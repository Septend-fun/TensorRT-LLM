# SPDX-FileCopyrightText: Copyright (c) 2026 The TensorRT-LLM Contributors
# SPDX-FileCopyrightText: Copyright 2026 The XHToken team and the HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.quantization.mode import QuantAlgo

from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ..configs.spark import Spark2_6Config
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


def gelu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.gelu(gate, approximate="none") * up


class SparkAttention(Attention):

    def __init__(self, model_config: ModelConfig[Spark2_6Config], layer_idx: int):
        config = model_config.pretrained_config
        layer_type = config.layer_types[layer_idx]
        layer_rope = config.rope_parameters_for_layer(layer_type)
        partial_rotary_factor = layer_rope.get("partial_rotary_factor", 1.0)
        rope_params = RopeParams(
            dim=int(config.head_dim * partial_rotary_factor),
            theta=layer_rope.get("rope_theta", 10000.0),
            max_positions=config.max_position_embeddings,
            original_max_positions=config.max_position_embeddings,
        )
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            dense_bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            head_dim=config.head_dim,
        )
        self.attention_window_size = (
            config.sliding_window if layer_type == "sliding_attention" else None)
        self.g_proj = None
        if config.headwise_attn_output_gate:
            self.g_proj = Linear(
                config.hidden_size,
                config.num_attention_heads,
                bias=config.attention_bias,
                dtype=config.torch_dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                reduce_output=False,
            )

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        output_gate = None
        if self.g_proj is not None:
            output_gate = self.g_proj(hidden_states).unsqueeze(-1)
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_window_size=self.attention_window_size,
            output_gate=output_gate,
            **kwargs,
        )


class SparkDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[Spark2_6Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.enable_allreduce = model_config.mapping.tp_size > 1
        self.self_attn = SparkAttention(model_config, layer_idx)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation=gelu_mul,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=self.enable_allreduce),
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=self.enable_allreduce),
            **kwargs,
        )
        return hidden_states, residual


class SparkModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Spark2_6Config]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            SparkDecoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden_states = (self.embed_tokens(input_ids)
                         if inputs_embeds is None else inputs_embeds)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Spark2_6ForCausalLM")
class Spark2_6ForCausalLM(DecoderModelForCausalLM[SparkModel, Spark2_6Config]):

    def __init__(self, model_config: ModelConfig[Spark2_6Config]):
        if model_config.mapping.cp_size > 1:
            raise ValueError(
                "Spark2_6ForCausalLM phase 1 does not support context parallelism"
            )
        quant_config = model_config.get_quant_config()
        if (quant_config is not None and quant_config.quant_algo is not None
                and quant_config.quant_algo != QuantAlgo.MIXED_PRECISION):
            raise ValueError(
                "Spark2_6ForCausalLM supports BF16 or the native Spark mixed-precision checkpoint")
        config = model_config.pretrained_config
        super().__init__(
            SparkModel(model_config),
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )

    def load_weights(self, weights, weight_mapper=None, **kwargs) -> None:
        # The generic v2 loader delegates per-module mapping but does not run
        # whole-checkpoint preprocessing. Spark needs that pass to translate
        # compressed-tensors packed INT4 and split its fused QKV projection.
        if weight_mapper is not None:
            weights = weight_mapper.preprocess_weights(weights)
        return super().load_weights(weights, weight_mapper=weight_mapper,
                                    **kwargs)


# Backward-compatible architecture spelling in the published 2.6 checkpoint.
Spark2_5ForCausalLM = Spark2_6ForCausalLM
register_auto_model("Spark2_5ForCausalLM")(Spark2_5ForCausalLM)


__all__ = ["Spark2_6ForCausalLM", "Spark2_5ForCausalLM", "SparkAttention", "SparkModel"]
