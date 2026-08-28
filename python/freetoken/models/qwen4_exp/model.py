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
from .vision import Qwen4VisionModel

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
        self._image_token_id = config.image_token_id
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

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scatter precomputed image soft-token embeddings at image-token positions.

        Same contract as Gemma4's ``_merge_multimodal`` (``models/gemma4/model.py:68-86``):
        ``mm_embeds`` (set by the scheduler from each request's vision features, see
        ``scheduler/scheduler.py``'s ``_gather_multimodal``) is a
        ``[num_image_tokens, hidden]`` tensor whose rows replace the placeholder embeddings
        produced for ``image_token_id``. Only runs during prefill batches that carry images;
        text-only batches (``mm_embeds is None``, the production default) return ``x``
        unchanged.

        Must be called on ``x`` BEFORE the hyper-connection replication
        (``x4 = x.repeat(1, self._hc_count)`` in ``forward`` below) -- splicing after the
        repeat would scatter each image embedding into only one of the ``hc_count``
        replicated copies instead of all of them, corrupting every hyper-connection stream
        but the first.
        """
        batch = get_global_ctx().batch
        mm_embeds = getattr(batch, "mm_embeds", None)
        if mm_embeds is None or self._image_token_id is None:
            return x
        mask = input_ids == self._image_token_id
        n_slots = int(mask.sum().item())
        assert n_slots == mm_embeds.shape[0], (
            f"image-token slots ({n_slots}) != vision features ({mm_embeds.shape[0]}); "
            "image tokens must not be split across prefill chunks"
        )
        return x.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(x.dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = self._merge_multimodal(input_ids, x)
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
        if config.is_multimodal:
            self.vision_tower = Qwen4VisionModel(config.vision_config)
        super().__init__()

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run the vision tower + merger. Returns ``[num_valid_soft_tokens, hidden_size]``.

        Unlike Gemma4, the merger already projects to the text hidden size, so there is no
        separate multimodal-embedder stage -- ``vision_tower.forward`` is the whole thing.

        ``pixel_values``: ``[num_images, num_patches, in_channels*temporal_patch_size*patch_size**2]``;
        ``image_position_ids``: ``[num_images, num_patches, 2]`` raw ``(row, col)`` patch-grid
        coordinates (0-indexed, pre-merge resolution), ``(-1, -1)`` padding. See
        ``Qwen4VisionModel`` (vision.py) for the full contract.
        """
        return self.vision_tower.forward(pixel_values, image_position_ids)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)

    def precompute_ngram_embed(self, batch) -> None:
        """Run every PLE layer's host-mmap n-gram gather eagerly, before ``forward()``
        may be dispatched through a captured CUDA graph. See
        ``Qwen4PLELayer.precompute_decode_ngram`` for why this must happen outside
        graph capture. Called by the engine for every decode step (captured or
        eager) -- a no-op if this checkpoint has no PLE layers."""
        for layer in self.model.layers.op_list:
            if layer.ple is not None:
                layer.ple.precompute_decode_ngram(batch)


__all__ = ["Qwen4ExpForCausalLM"]
