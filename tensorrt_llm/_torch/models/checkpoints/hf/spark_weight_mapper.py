# SPDX-FileCopyrightText: Copyright (c) 2026 The TensorRT-LLM Contributors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

import torch

from tensorrt_llm._torch.models.modeling_utils import register_mapper

from ..base_weight_loader import ConsumableWeightsDict
from .weight_mapper import HfWeightMapper


@register_mapper("HF", "Spark2_6ForCausalLM")
@register_mapper("HF", "Spark2_5ForCausalLM")
class SparkHfWeightMapper(HfWeightMapper):

    @staticmethod
    def _unpack_int4(packed: torch.Tensor, shape: tuple[int, int],
                     packed_dim: int = 1, signed: bool = True) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        values = ((packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF)
        if signed:
            values = torch.where(values >= 8, values - 16, values)
        values = values.to(torch.int8)
        if packed_dim == 1:
            values = values.reshape(shape[0], -1)
        else:
            values = values.movedim(-1, 1).reshape(-1, shape[1])
        return values[:shape[0], :shape[1]].contiguous()

    @staticmethod
    def _pack_kernel_int4(values: torch.Tensor) -> torch.Tensor:
        return values.contiguous()

    def _convert_w4(self, weights: Mapping[str, torch.Tensor],
                    prefix: str) -> dict[str, torch.Tensor]:
        shape = tuple(int(v) for v in weights[prefix + ".weight_shape"][:].tolist())
        n, k = shape
        scale = weights[prefix + ".weight_scale"][:].to(torch.bfloat16)
        if tuple(scale.shape) != (n, k // 32):
            raise ValueError(f"Unexpected Spark W4 scale shape for {prefix}: {scale.shape}")
        # Spark's group-quantized weights are asymmetric INT4: both the
        # stored nibble and zero-point use the unsigned [0, 15] code domain.
        codes = self._unpack_int4(
            weights[prefix + ".weight_packed"][:], shape, signed=False)
        zero_points = self._unpack_int4(
            weights[prefix + ".weight_zero_point"][:], (n, k // 32),
            packed_dim=0, signed=False)
        return {
            "weight": self._pack_kernel_int4(codes),
            "weight_scale": scale,
            "zero": zero_points.to(torch.bfloat16).contiguous(),
        }

    def preprocess_weights(
        self, weights: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        config = self.model.config
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        renamed: dict[str, torch.Tensor] = {}

        w4_prefixes = [
            name.removesuffix(".weight_packed") for name in weights
            if name.endswith(".weight_packed")
        ]
        consumed = set()
        for prefix in w4_prefixes:
            converted = self._convert_w4(weights, prefix)
            consumed.update({
                prefix + ".weight_packed", prefix + ".weight_scale",
                prefix + ".weight_shape", prefix + ".weight_zero_point"
            })
            marker = ".self_attn.q_k_v_proj"
            if marker in prefix:
                target_prefix = prefix.replace(marker, ".self_attn")
                for parameter, tensor in converted.items():
                    sizes = ((q_size, kv_size, kv_size)
                             if parameter == "weight"
                             else (q_size, kv_size, kv_size))
                    q, k, v = tensor.split(sizes, dim=0)
                    renamed[f"{target_prefix}.q_proj.{parameter}"] = q
                    renamed[f"{target_prefix}.k_proj.{parameter}"] = k
                    renamed[f"{target_prefix}.v_proj.{parameter}"] = v
            else:
                for parameter, tensor in converted.items():
                    renamed[f"{prefix}.{parameter}"] = tensor

        for name, tensor in weights.items():
            if name in consumed:
                continue
            name = name.replace("model.embedding.", "model.embed_tokens.")
            name = name.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            marker = ".self_attn.q_k_v_proj."
            if marker not in name:
                renamed[name] = tensor
                continue

            prefix, parameter = name.split(marker, maxsplit=1)
            if tensor.shape[0] != q_size + 2 * kv_size:
                raise ValueError(
                    f"Unexpected fused Spark QKV shape for {name}: {tuple(tensor.shape)}")
            q, k, v = tensor.split((q_size, kv_size, kv_size), dim=0)
            renamed[f"{prefix}.self_attn.q_proj.{parameter}"] = q
            renamed[f"{prefix}.self_attn.k_proj.{parameter}"] = k
            renamed[f"{prefix}.self_attn.v_proj.{parameter}"] = v
        return ConsumableWeightsDict.take_ownership(weights, renamed)


__all__ = ["SparkHfWeightMapper"]
