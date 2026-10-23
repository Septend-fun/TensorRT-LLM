# SPDX-FileCopyrightText: Copyright (c) 2026 The TensorRT-LLM Contributors
# SPDX-FileCopyrightText: Copyright 2026 The XHToken team and the HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0

from transformers.configuration_utils import PretrainedConfig


class Spark2_6Config(PretrainedConfig):
    model_type = "spark2_6"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        headwise_attn_output_gate: bool = False,
        gate_attn_act_mode: str = "sigmoid",
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        **kwargs,
    ):
        num_key_value_heads = num_key_value_heads or num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if hidden_act != "gelu":
            raise ValueError("Spark2_6Config only supports hidden_act='gelu'")
        if gate_attn_act_mode != "sigmoid":
            raise ValueError("Spark2_6Config only supports sigmoid attention gates")

        layer_types = layer_types or ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")
        unsupported = set(layer_types) - {"full_attention", "sliding_attention"}
        if unsupported:
            raise ValueError(f"Unsupported Spark attention layer types: {sorted(unsupported)}")
        if "sliding_attention" in layer_types and not sliding_window:
            raise ValueError("sliding_window must be set when sliding layers are present")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.rope_parameters = rope_parameters or {
            "full_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            "sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        }
        super().__init__(**kwargs)

    def rope_parameters_for_layer(self, layer_type: str) -> dict:
        return self.rope_parameters.get(layer_type, {})


__all__ = ["Spark2_6Config"]
