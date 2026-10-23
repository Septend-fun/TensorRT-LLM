# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Mapping, Optional

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def update_quant_config_from_compressed_tensors(
    quant_config: QuantConfig, hf_quant_config: Mapping[str, Any]
) -> Optional[dict[str, QuantConfig]]:
    """Mutate QuantConfig from an llm-compressor compressed-tensors config."""
    config_groups = hf_quant_config.get("config_groups")
    if config_groups is None:
        raise ValueError(f"config_groups is not set in {hf_quant_config}.")

    if hf_quant_config.get("format") == "mixed-precision":
        quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
        layer_configs: dict[str, QuantConfig] = {}
        for group_name, group in config_groups.items():
            weights = group["weights"]
            inputs = group.get("input_activations")
            if (group.get("format") == "pack-quantized"
                    and weights.get("num_bits") == 4
                    and weights.get("type") == "int"
                    and weights.get("strategy") == "group"
                    and inputs is None):
                if weights.get("group_size") != 32:
                    raise ValueError(
                        f"Unsupported INT4 group_size in {group_name}: "
                        f"{weights.get('group_size')}; supported: 32.")
                layer_config = QuantConfig(
                    quant_algo=QuantAlgo.W4A16_AWQ,
                    group_size=32,
                    has_zero_point=not weights.get("symmetric", True),
                    pre_quant_scale=False,
                )
            elif (group.get("format") == "int-quantized"
                  and weights.get("num_bits") == 8
                  and weights.get("type") == "int"
                  and weights.get("strategy") == "channel"
                  and inputs is not None
                  and inputs.get("num_bits") == 8
                  and inputs.get("type") == "int"
                  and inputs.get("strategy") == "token"
                  and inputs.get("dynamic") is True):
                layer_config = QuantConfig(
                    quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)
            else:
                raise ValueError(
                    f"Unsupported mixed-precision group {group_name}: {group}.")

            for target in group.get("targets", []):
                # compressed-tensors commonly uses anchored regexes. Preserve
                # the useful module suffix as a glob consumed by the PyTorch
                # layerwise quantization pass.
                target = target.removeprefix("re:").removesuffix("$")
                target = target.replace(".*", "*")
                target = target.replace("(gate_proj|up_proj|gate_up_proj|down_proj)", "*")
                target = target.replace("(gate_proj|up_proj|gate_up_proj)", "*")
                if target.startswith("*"):
                    target = "model.layers.*." + target[1:]
                target = target.replace("q_k_v_proj", "q_proj")
                target = target.replace("out_proj", "o_proj")
                layer_configs[target] = layer_config

        quant_config.exclude_modules = list(hf_quant_config.get("ignore", []))
        return layer_configs

    # compressed-tensors keys config_groups by group name: custom recipes use
    # "group_0"; named preset schemes (e.g. FP8_DYNAMIC) use the scheme name.
    # This path applies a single algo globally, so resolve to one group.
    group_config = config_groups.get("group_0")
    if group_config is None:
        if len(config_groups) != 1:
            raise ValueError(
                f"Expected 'group_0' or exactly one config group, got {sorted(config_groups)}."
            )
        group_config = next(iter(config_groups.values()))
    weights_quant_config = group_config["weights"]
    weights_quant_strategy = weights_quant_config["strategy"]

    # kv_cache_scheme (llm-compressor): FP8 per-tensor KV cache. Handled
    # before the weight-algo branches so recipes that early-return (MXFP4
    # pack-quantized) still pick up the KV-cache quantization.
    kv_cache_scheme = hf_quant_config.get("kv_cache_scheme")
    if kv_cache_scheme is not None:
        if kv_cache_scheme.get("num_bits") == 8 and kv_cache_scheme.get("type") == "float":
            if quant_config.kv_cache_quant_algo in (None, QuantAlgo.FP8):
                quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            else:
                raise ValueError(
                    f"Specified kv_cache_quant_algo={quant_config.kv_cache_quant_algo}, "
                    "conflicting with FP8 KV cache from HF quant config."
                )
        else:
            raise ValueError(f"Unsupported kv_cache_scheme: {kv_cache_scheme}.")

    # MXFP4 pack-quantized (weight-only): FP4 E2M1 weights packed two per
    # uint8 with per-32-group uint8 E8M0 scales and no activation
    # quantization (e.g. Kimi K3 routed experts). Handled before reading the
    # input-activation strategy, which is null for weight-only recipes.
    if hf_quant_config.get("format") == "mxfp4-pack-quantized" or (
        weights_quant_config["num_bits"] == 4
        and weights_quant_config.get("type") == "float"
        and weights_quant_strategy == "group"
        and group_config.get("input_activations") is None
    ):
        group_size = weights_quant_config["group_size"]
        if group_size != 32:
            raise ValueError(f"Unsupported group_size: {group_size}. Supported: 32 for MXFP4.")
        quant_config.quant_algo = QuantAlgo.W4A16_MXFP4
        quant_config.group_size = group_size
        hf_exclude_modules = hf_quant_config.get("modules_to_not_convert", None)
        quant_config.exclude_modules = list(
            set((hf_exclude_modules or []) + hf_quant_config.get("ignore", []))
        )
        return None

    inputs_quant_config = group_config["input_activations"]
    inputs_quant_strategy = inputs_quant_config["strategy"]

    if weights_quant_config["num_bits"] == 8:
        if weights_quant_strategy == "channel":
            if inputs_quant_strategy != "token":
                raise ValueError(f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}.")
            quant_config.quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        elif weights_quant_strategy == "block":
            if inputs_quant_strategy != "group":
                raise ValueError(f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}.")
            quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
            group_size = inputs_quant_config["group_size"]

            # TRT-LLM only supports group_size=128 for FP8_BLOCK_SCALES.
            if group_size != 128:
                raise ValueError(f"Unsupported group_size: {group_size}. Supported: 128.")
            quant_config.group_size = group_size

        else:
            raise ValueError(
                f"Unsupported weights_quant_strategy: {weights_quant_strategy}. "
                "Supported strategies: 'channel', 'block'."
            )
    elif (
        weights_quant_config["num_bits"] == 4
        and weights_quant_config.get("type") == "float"
        and weights_quant_strategy == "tensor_group"
    ):
        # llm-compressor NVFP4: weights FP4 with FP8 per-group scales
        # (group_size=16), scaled by an FP32 global scale.
        if inputs_quant_strategy != "tensor_group":
            raise ValueError(
                f"Unsupported inputs_quant_strategy for NVFP4: {inputs_quant_strategy}."
            )
        group_size = weights_quant_config["group_size"]
        if group_size != 16:
            raise ValueError(f"Unsupported group_size: {group_size}. Supported: 16 for NVFP4.")
        quant_config.quant_algo = QuantAlgo.NVFP4
        quant_config.group_size = group_size
    else:
        raise ValueError(
            f"Unsupported quant_bits: {weights_quant_config['num_bits']}. "
            "Supported: 8 (FP8) or 4 (NVFP4)."
        )

    hf_exclude_modules = hf_quant_config.get("modules_to_not_convert", None)
    if hf_exclude_modules is not None:
        quant_config.exclude_modules = list(
            set(hf_exclude_modules + hf_quant_config.get("ignore", []))
        )
    else:
        quant_config.exclude_modules = hf_quant_config.get("ignore", [])
    return None
