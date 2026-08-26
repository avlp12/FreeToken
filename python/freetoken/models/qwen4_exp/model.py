from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .gdn import Qwen4GatedDeltaNet
from .hyperconnect import Qwen4GatedResidual
from .moe import Qwen4ExpMoE
from .ngram import Qwen4PLELayer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Assembly mirrors transformers modeling_qwen4_exp exactly:
#   x4 = embed(ids).repeat(1, hc_count)              # 4-wide residual streams
#   per layer:  x4 += ple(x4, ids)                   # PLE layers only (layer 1)
#               mixed, w = attn_hc(x4); h = mixer(mixed); x4 += h (x) w
#               mixed, w = mlp_hc(x4);  h = moe(mixed);   x4 += h (x) w
#   out = hyper_connection_mixer.mix(x4)             # final collapse (doubles as norm)
#   logits = lm_head(out)                            # untied
# There are NO input/post_attention layernorms and NO final model.norm in this
# architecture -- the hyper-connection grouped norms play those roles (checkpoint has
# no such keys; verified against the shipped index.json).


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.qwen4_args
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen4GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                # Dense GDN weights are bf16 in the FP8 release (whitelisted): never
                # route them through the fp8_block projections qwen3_5 would pick.
                expert_quant="none",
                attn_quant="none",
                output_gate_type=args.output_gate_type,
            )
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps
        )
        self.mlp_hyper_connection = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps
        )
        self.ple = (
            Qwen4PLELayer(config, layer_id)
            if layer_id in args.ple_layer_indices
            else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            x4 = x4 + self.ple.forward(x4)

        mixed, inject = self.attn_hyper_connection.forward(x4)
        h = self.linear_attn.forward(mixed) if self._is_linear else self.self_attn.forward(mixed)
        x4 = self.attn_hyper_connection.combine(x4, h, inject)

        mixed, inject = self.mlp_hyper_connection.forward(x4)
        h = self.mlp.forward(mixed)
        x4 = self.mlp_hyper_connection.combine(x4, h, inject)
        return x4


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        args = config.qwen4_args
        self._hc_count = args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps,
            use_combine=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x4 = x.repeat(1, self._hc_count)
        for layer in self.layers.op_list:
            x4 = layer.forward(x4)
        return self.hyper_connection_mixer.mix(x4)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen4ExpModel(config)
        assert not config.tie_word_embeddings, "Qwen3.8-Flash-Next ships an untied lm_head"
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=False,
            tied_embedding=None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen4ExpForCausalLM"]
