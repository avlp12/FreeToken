from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    vision_load_enabled,
)


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen4-Exp (Qwen3.8-Flash-Next) extras the generic ModelConfig has no fields for.

    Carried opaquely on ``ModelConfig.qwen4_args`` (same pattern as dsv4_args/m3_args).
    P0 serves the QSA layers as dense full attention -- exact for sequences up to
    ``indexer_budget`` because the indexer selects *all* visible tokens until more than
    ``indexer_budget`` complete compressed blocks exist. The indexer geometry is parsed
    and stored now so the P1 sparse backend reads the same source of truth.
    """

    # QSA indexer (dormant in P0; weights are loaded so the FTW stays complete)
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # Hyper-connections (mHC low-rank)
    hc_count: int = 4
    hc_lowrank: int = 320
    # n-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_layer_ids: tuple[int, ...] = (2,)  # ONE-indexed (HF convention)
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    seed: int = 1234
    # GDN output gate activation ("sigmoid" for Qwen3.8; qwen3_5 used silu)
    output_gate_type: str = "sigmoid"
    # PLE needs the eos id for n-gram context resets / fresh-state fill
    eos_token_id: int = 248044
    # Checkpoint dir: the n-gram table (~51 GB fp8) is NOT streamed through iter_weights;
    # the PLE module maps the shards straight from here (host-resident, lazy).
    model_path: str | None = None
    # MTP drafter config, preserved for a later phase (dropped from loading in P0).
    mtp: dict = field(default_factory=dict)
    # --expert-gguf: set post-parse (object.__setattr__, this dataclass is frozen) by
    # engine.engine._apply_expert_gguf, the same seam DeepseekV4Args.expert_gguf_path
    # uses. Read by moe.expert_banks._q4_k_ud_banks.
    expert_gguf_path: str | None = None

    @property
    def ple_layer_indices(self) -> tuple[int, ...]:
        """Zero-based decoder layer indices that own a PLE block.

        HF's ``ple_layer_ids`` are ONE-indexed: the reference decoder attaches PLE when
        ``layer_idx + 1 in config.ple_layer_ids`` (modeling_qwen4_exp.py), and the released
        checkpoint stores the PLE weights under ``layers.1.ple.*`` with ple_layer_ids=[2].
        """
        return tuple(i - 1 for i in self.ple_layer_ids)


@dataclass(frozen=True)
class VisionConfig:
    """Qwen4-Exp's vision tower config (`python/freetoken/models/qwen4_exp/vision.py`).

    Field names follow the checkpoint's `vision_config` (== HF's `Qwen3VLVisionConfig`
    schema -- this vision tower's weights are tensor-for-tensor identical to Qwen3-VL's,
    see vision.py's module docstring), not Gemma4's `VisionConfig` shape: no `num_kv_heads`
    (plain MHA, no GQA), `temporal_patch_size` instead of a single spatial-only patch,
    `num_position_embeddings`/`out_hidden_size` instead of
    `position_embedding_size`/relying on `text_hidden_size` for the projector.
    """

    hidden_size: int
    num_layers: int  # HF's "depth"
    num_heads: int
    head_dim: int
    intermediate_size: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    hidden_act: str
    rope_theta: float
    text_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...] = ()


def _parse_vision_config(hf_config: Any, text_hidden_size: int) -> VisionConfig | None:
    vc = getattr(hf_config, "vision_config", None)
    if vc is None:
        return None
    # Vision is opt-in (default OFF), same switch Gemma4 uses (`FREETOKEN_LOAD_VISION=1`):
    # is_multimodal flows through parse_config into both model build and weight loading, so
    # returning None here keeps default `ft serve` boot byte-for-byte unchanged. Note the
    # production FTW checkpoint has no `model.visual.*` tensors at all yet (dropped by
    # checkpoint/qwen_layout.py's SKIP_PREFIXES) -- opting in today will fail to load until
    # the converter is extended separately; this wiring only makes the *module* reachable.
    if not vision_load_enabled():
        return None
    num_heads = int(vc.num_heads)
    return VisionConfig(
        hidden_size=int(vc.hidden_size),
        num_layers=int(vc.depth),
        num_heads=num_heads,
        head_dim=int(vc.hidden_size) // num_heads,
        intermediate_size=int(vc.intermediate_size),
        in_channels=int(getattr(vc, "in_channels", 3)),
        patch_size=int(vc.patch_size),
        temporal_patch_size=int(vc.temporal_patch_size),
        spatial_merge_size=int(vc.spatial_merge_size),
        num_position_embeddings=int(vc.num_position_embeddings),
        out_hidden_size=int(vc.out_hidden_size),
        hidden_act=str(getattr(vc, "hidden_act", "gelu_pytorch_tanh")),
        # Qwen3VLVisionConfig carries no rope_theta field; the HF rotary module hardcodes
        # this default (Qwen3VLVisionRotaryEmbedding.__init__'s theta=10000.0).
        rope_theta=10000.0,
        text_hidden_size=text_hidden_size,
        deepstack_visual_indexes=tuple(getattr(vc, "deepstack_visual_indexes", None) or ()),
    )


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def _fp8_block_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """Qwen3.8-Flash-Next-FP8: quant_method=fp8, weight_block_size=[128,128], and the
    ``modules_to_not_convert`` whitelist covers EVERYTHING except the routed experts
    (verified against the shipped index.json: only ``mlp.experts.*`` tensors carry a
    ``weight_scale_inv``). So unlike qwen3_5-FP8, ``fp8_block`` here means "experts only";
    all dense modules in this package are built bf16 unconditionally."""
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return "none", None
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or get("quant_algo") or "").lower()
    block = get("weight_block_size")
    if method == "fp8" and block:
        bs = tuple(int(x) for x in block)
        assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
        return "fp8_block", bs
    return "none", None


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = round(head_dim * partial)
    # Text-only serving: with all three mRoPE position rows equal, interleaved-mRoPE
    # cos/sin degenerate to standard partial NeoX rope, so no scaling dict is carried
    # (same reduction qwen3_5_moe ships; mrope_section is unhashable for the rope cache).
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _fp8_block_quant(hf_config)

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t != "linear_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate=True,
    )
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    eos = getattr(text, "eos_token_id", 248044)
    if isinstance(eos, list):
        eos = eos[0]

    qwen4_args = Qwen4ExpArgs(
        indexer_n_heads=int(getattr(text, "indexer_n_heads", 4) or 4),
        indexer_kv_heads=int(getattr(text, "indexer_kv_heads", 1) or 1),
        indexer_head_dim=int(getattr(text, "indexer_head_dim", 128) or 128),
        indexer_budget=int(getattr(text, "indexer_budget", 2048) or 2048),
        indexer_compress_ratio=int(getattr(text, "indexer_compress_ratio", 4) or 4),
        hc_count=int(getattr(text, "hc_count", 4)),
        hc_lowrank=int(getattr(text, "hc_lowrank", 320)),
        ngram_size=int(getattr(text, "ngram_size", 3)),
        heads_per_ngram=int(getattr(text, "heads_per_ngram", 8)),
        ngram_vocab_size_base=int(getattr(text, "ngram_vocab_size_base", 20_000_000)),
        make_ngram_vocab_size_divisible_by=int(
            getattr(text, "make_ngram_vocab_size_divisible_by", 128)
        ),
        split_ngram_parts=int(getattr(text, "split_ngram_parts", 128)),
        ple_layer_ids=tuple(getattr(text, "ple_layer_ids", None) or ()),
        ple_embed_dim=int(getattr(text, "ple_embed_dim", None) or text.hidden_size),
        ple_conv_kernel_size=int(getattr(text, "ple_conv_kernel_size", 4)),
        seed=int(getattr(text, "seed", 1234)),
        output_gate_type=str(
            getattr(text, "output_gate_type", None) or getattr(text, "hidden_act", "silu")
        ),
        eos_token_id=int(eos),
        model_path=getattr(hf_config, "_name_or_path", None) or None,
        mtp=dict(getattr(text, "mtp", None) or {}),
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0) or 0,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=int(getattr(text, "num_experts", 0) or 0),
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        shared_expert_intermediate_size=int(
            getattr(text, "shared_expert_intermediate_size", 0) or 0
        ),
        # transformers Qwen4ExpTextConfig defaults norm_topk_prob to TRUE (checkpoint
        # config.json omits the field) -- differs from the qwen3_5 default of False.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=True,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=_parse_vision_config(hf_config, text.hidden_size),
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        # Dense modules (attention, GDN, shared expert, HC, PLE, lm_head) are bf16 in the
        # released FP8 checkpoint -- the fp8_block expert_quant applies to banks only.
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        qwen4_args=qwen4_args,
    )


__all__ = ["parse_config", "Qwen4ExpArgs", "VisionConfig"]
