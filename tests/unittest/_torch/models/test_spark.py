# SPDX-FileCopyrightText: Copyright (c) 2026 The TensorRT-LLM Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.configs.spark import Spark2_6Config
from tensorrt_llm._torch.models.checkpoints.auto_mapper import \
    AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.spark_weight_mapper import \
    SparkHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class


def test_spark_config_validates_layer_schedule() -> None:
    config = Spark2_6Config(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=512,
    )
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert config.model_type == "spark2_6"

    with pytest.raises(ValueError, match="one entry per decoder layer"):
        Spark2_6Config(num_hidden_layers=2, layer_types=["full_attention"])


def test_spark_model_is_available_through_lazy_registry() -> None:
    model_class = get_registered_model_class("Spark2_6ForCausalLM")
    assert model_class is not None
    assert model_class.__name__ == "Spark2_6ForCausalLM"


def test_spark_mapper_is_available_through_auto_registry() -> None:
    mapper = AutoCheckpointMapper.get("HF", "Spark2_6ForCausalLM")
    assert isinstance(mapper, SparkHfWeightMapper)


def test_spark_mapper_splits_fused_qkv_and_renames_modules() -> None:
    config = Spark2_6Config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
    )
    mapper = SparkHfWeightMapper()
    mapper._model = SimpleNamespace(config=config)
    fused = torch.arange(16 * 8).reshape(16, 8)
    mapped = mapper.preprocess_weights({
        "model.embedding.weight": torch.empty(32, 8),
        "model.layers.0.self_attn.q_k_v_proj.weight": fused,
        "model.layers.0.self_attn.out_proj.weight": torch.empty(8, 8),
    })

    assert mapped["model.layers.0.self_attn.q_proj.weight"].shape == (8, 8)
    assert mapped["model.layers.0.self_attn.k_proj.weight"].shape == (4, 8)
    assert mapped["model.layers.0.self_attn.v_proj.weight"].shape == (4, 8)
    assert "model.embed_tokens.weight" in mapped
    assert "model.layers.0.self_attn.o_proj.weight" in mapped


def test_spark_mapper_keeps_asymmetric_int4_codes_unsigned() -> None:
    packed = torch.tensor([[-2023406815]], dtype=torch.int32)  # 0x87654321
    unsigned = SparkHfWeightMapper._unpack_int4(packed, (1, 8), signed=False)
    signed = SparkHfWeightMapper._unpack_int4(packed, (1, 8), signed=True)
    assert unsigned.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert signed.tolist() == [[1, 2, 3, 4, 5, 6, 7, -8]]


def test_spark_mapper_preserves_consumable_weight_ownership() -> None:
    config = Spark2_6Config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
    )
    mapper = SparkHfWeightMapper()
    mapper._model = SimpleNamespace(config=config)
    source = ConsumableWeightsDict({
        "model.layers.0.self_attn.q_k_v_proj.weight": torch.empty(16, 8),
    })

    mapped = mapper.preprocess_weights(source)

    assert isinstance(mapped, ConsumableWeightsDict)
    assert len(source) == 0
    assert len(mapped) == 3


def test_spark_mapper_converts_packed_asymmetric_w4_qkv() -> None:
    config = Spark2_6Config(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
    )
    mapper = SparkHfWeightMapper()
    mapper._model = SimpleNamespace(config=config)
    shape = (32, 32)
    codes = (torch.arange(shape[0] * shape[1], dtype=torch.int8).reshape(shape) % 16) - 8
    packed_bytes = ((codes.to(torch.uint8)[:, ::2] & 15)
                    | ((codes.to(torch.uint8)[:, 1::2] & 15) << 4))
    packed_i32 = packed_bytes.contiguous().view(torch.int32)
    zeros = torch.zeros((shape[0], 1), dtype=torch.int8)
    zeros_packed = torch.zeros((shape[0] // 8, 1), dtype=torch.int32)
    scale = torch.ones((shape[0], 1), dtype=torch.bfloat16)
    prefix = "model.layers.0.self_attn.q_k_v_proj"

    mapped = mapper.preprocess_weights({
        prefix + ".weight_packed": packed_i32,
        prefix + ".weight_scale": scale,
        prefix + ".weight_shape": torch.tensor(shape),
        prefix + ".weight_zero_point": zeros_packed,
    })

    assert mapped["model.layers.0.self_attn.q_proj.weight"].shape == (16, 32)
    assert mapped["model.layers.0.self_attn.k_proj.weight_scale"].shape == (8, 1)
    restored = mapper._unpack_int4(packed_i32, shape)
    torch.testing.assert_close(restored, codes)


def test_attention_external_gate_broadcasts_per_head() -> None:
    attention = SimpleNamespace(head_dim=2)
    output = torch.ones(3, 4)
    gate = torch.zeros(3, 2, 1)
    gated = Attention.apply_output_gate(attention, output, gate)
    torch.testing.assert_close(gated, torch.full_like(output, 0.5))
