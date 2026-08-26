"""Qwen3.8-Flash-Next MTP drafter skeleton.

Standalone module. Do **not** import ``freetoken.models.qwen4_exp`` body modules from
here -- that package is being written in parallel. Binding points are named in
comments and in :data:`BODY_BINDING_POINTS`.

Official HuggingFace ``transformers`` ``qwen4_exp`` (modeling + configuration,
fetched 2026-08-26 from ``main``) contains **no** MTP classes, no ``mtp`` config
fields, and no ``fc_embedding`` / ``fc_hidden``. Structure below is therefore
fixed from:

1. Checkpoint tensors under ``mtp.*`` (index + ``quantization_config.modules_to_not_convert``).
2. ``text_config.mtp`` / ``mtp_num_hidden_layers`` / ``mtp_use_dedicated_embeddings``.
3. DeepSeek-V3 / Qwen3-Next / Qwen3.5 MTP literature: norm(embed) and
   norm(hidden) fused, then one decoder block, then shared ``lm_head``.
4. This checkpoint **splits** the usual ``mtp.fc`` (``2H -> H``) into
   ``mtp.fc_embedding`` + ``mtp.fc_hidden``. Algebra of ``y = [e; h] @ W`` with
   ``W = [W_e | W_h]`` is ``y = e @ W_e^T + h @ W_h^T`` when each weight is
   ``[H, H]``. Other shapes raise :class:`NotImplementedError`.

Uncertain formulae raise ``NotImplementedError`` with the evidence in the
exception / adjacent comment. GPU execution is out of scope for this file.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

# ---------------------------------------------------------------------------
# Binding points for the parallel qwen4_exp body / engine (do not import them)
# ---------------------------------------------------------------------------

BODY_BINDING_POINTS: dict[str, str] = {
    "decoder_layer": (
        "Qwen4ExpTextDecoderLayer(text_config_with_mtp_layer_types, layer_idx=0). "
        "MTP layer_types is [full_attention]: GQA + QSA indexer + SparseMoE + "
        "attn/mlp hyper-connections. Official forward expects last-dim "
        "hc_count*hidden (repeat embeddings before the stack)."
    ),
    "hyper_connection_mixer": (
        "Qwen4ExpTextGatedResidual(config, use_combine=False). Checkpoint has "
        "hc_norm / input_mix_weight_{down,up} and NO block_inject_weight -- "
        "same as the body's final mixer. Output width is hidden_size (2560)."
    ),
    "rms_norm": (
        "Qwen4ExpTextRMSNorm. Body init is zero-centered (forward uses 1+weight). "
        "Inject the body's norm; do not re-derive the (1+w) vs w formula here."
    ),
    "embed_tokens": (
        "Target model.language_model.embed_tokens. "
        "mtp_use_dedicated_embeddings is False; no mtp.embed_tokens tensor."
    ),
    "lm_head": (
        "Shared target lm_head. No mtp.norm tensor (unlike Qwen3-Next/3.5 MTP). "
        "Mixer hc_norm is the last MTP-owned normalize."
    ),
    "hidden_tap": (
        "Target last_hidden after language_model.hyper_connection_mixer "
        "(width 2560). mtp_use_hidden_state_from_layer is null. Pre-mixer "
        "hc streams are 10240-wide and cannot feed fc_hidden."
    ),
    "engine_extra_moe_layers": (
        "ModelConfig.extra_moe_layers += num_hidden_layers when MTP is enabled, "
        "same as DSV4 n_draft_layers. Expert banks / offload cache index the "
        "MTP MoE as layer num_layers+k (48 for this checkpoint)."
    ),
    "engine_draft": (
        "Clone DeepseekV4ForCausalLM.draft / catch_up_draft_context. MTP drafts "
        "one token per step (Qwen MTP), not a dSpark block. KV is GQA+QSA, not MLA."
    ),
    "engine_flag": (
        "--speculative-dspark is DSV4-only. Add a sibling --speculative-mtp; "
        "do not reuse the dspark flag."
    ),
}


# Checkpoint key prefixes as they appear in model.safetensors.index.json
CKPT_PREFIX = "mtp."


@dataclass(frozen=True)
class QwenMtpConfig:
    """Parsed MTP section plus the text_config fields the drafter inherits."""

    # text_config.mtp
    hybrid: bool
    layer_types: tuple[str, ...]
    num_hidden_layers: int
    rope_theta: float
    mtp_use_hidden_state_from_layer: int | None
    # text_config top-level MTP flags
    mtp_use_dedicated_embeddings: bool
    # inherited body geometry (needed to size fusion / to document the block)
    hidden_size: int
    hc_count: int
    hc_lowrank: int
    rms_norm_eps: float
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    indexer_n_heads: int | None
    indexer_kv_heads: int | None
    indexer_head_dim: int | None
    indexer_budget: int | None
    indexer_compress_ratio: int | None
    partial_rotary_factor: float

    @property
    def draft_layer_ids(self) -> tuple[int, ...]:
        """Cache / expert-bank ids continuing the target stack (48, ...)."""
        # Target layer count is not stored here; the body passes it at bind time.
        raise NotImplementedError(
            "draft_layer_ids needs the target's num_hidden_layers from the body "
            "ModelConfig (expected 48 + k). Bind via QwenMtpDrafter.bind()."
        )


def _text_config(hf: Mapping[str, Any]) -> Mapping[str, Any]:
    text = hf.get("text_config")
    if isinstance(text, dict):
        return text
    return hf


def parse_mtp_config(hf_config: Mapping[str, Any] | Any) -> QwenMtpConfig:
    """Parse MTP knobs from a HF config mapping or object with the same fields."""
    if not isinstance(hf_config, Mapping):
        raw = {
            "text_config": getattr(hf_config, "text_config", None),
        }
        text_obj = raw["text_config"]
        if text_obj is not None and not isinstance(text_obj, Mapping):
            text = {k: getattr(text_obj, k, None) for k in dir(text_obj) if not k.startswith("_")}
        else:
            text = text_obj or {}
            if not isinstance(text, Mapping):
                text = {}
    else:
        text = _text_config(hf_config)

    mtp = text.get("mtp") or {}
    if not isinstance(mtp, Mapping):
        mtp = {}
    rope = text.get("rope_parameters") or {}
    if not isinstance(rope, Mapping):
        rope = {}

    layer_types = tuple(mtp.get("layer_types") or ("full_attention",))
    n_layers = int(mtp.get("num_hidden_layers") or text.get("mtp_num_hidden_layers") or 1)
    tap = mtp.get("mtp_use_hidden_state_from_layer")
    if tap is not None:
        tap = int(tap)

    return QwenMtpConfig(
        hybrid=bool(mtp.get("hybrid", False)),
        layer_types=layer_types,
        num_hidden_layers=n_layers,
        rope_theta=float(mtp.get("rope_theta") or rope.get("rope_theta") or 10_000_000),
        mtp_use_hidden_state_from_layer=tap,
        mtp_use_dedicated_embeddings=bool(text.get("mtp_use_dedicated_embeddings", False)),
        hidden_size=int(text["hidden_size"]),
        hc_count=int(text.get("hc_count", 4)),
        hc_lowrank=int(text.get("hc_lowrank", 320)),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        vocab_size=int(text["vocab_size"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        num_experts=int(text["num_experts"]),
        num_experts_per_tok=int(text["num_experts_per_tok"]),
        moe_intermediate_size=int(text["moe_intermediate_size"]),
        shared_expert_intermediate_size=int(text.get("shared_expert_intermediate_size", 0)),
        indexer_n_heads=text.get("indexer_n_heads"),
        indexer_kv_heads=text.get("indexer_kv_heads"),
        indexer_head_dim=text.get("indexer_head_dim"),
        indexer_budget=text.get("indexer_budget"),
        indexer_compress_ratio=text.get("indexer_compress_ratio"),
        partial_rotary_factor=float(
            rope.get("partial_rotary_factor") or text.get("partial_rotary_factor") or 0.25
        ),
    )


# Weight names as stored in the Flash-Next-FP8 index (no body remapping).
MTP_WEIGHT_NAMES: dict[str, str] = {
    "fc_embedding": "mtp.fc_embedding.weight",
    "fc_hidden": "mtp.fc_hidden.weight",
    "pre_fc_norm_embedding": "mtp.pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden": "mtp.pre_fc_norm_hidden.weight",
    "mixer_hc_norm": "mtp.hyper_connection_mixer.hc_norm.weight",
    "mixer_mix_down": "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mixer_mix_up": "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
}

MTP_LAYER0_PREFIXES: tuple[str, ...] = (
    "mtp.layers.0.self_attn.",
    "mtp.layers.0.mlp.",
    "mtp.layers.0.attn_hyper_connection.",
    "mtp.layers.0.mlp_hyper_connection.",
)

# Expert bank: 512 routed experts, FP8 block-scaled (weight + weight_scale_inv).
# Shared expert / gate / attn / hyper-connections are in modules_to_not_convert (BF16).
MTP_EXPERT_BANK_KEYS = (
    "mtp.layers.0.mlp.experts.{i}.gate_proj.weight",
    "mtp.layers.0.mlp.experts.{i}.up_proj.weight",
    "mtp.layers.0.mlp.experts.{i}.down_proj.weight",
)


def mtp_expert_weight_names(num_experts: int = 512) -> list[str]:
    names: list[str] = []
    for i in range(num_experts):
        for tmpl in MTP_EXPERT_BANK_KEYS:
            names.append(tmpl.format(i=i))
            names.append(tmpl.format(i=i).replace(".weight", ".weight_scale_inv"))
    return names


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse a safetensors JSON header (no tensor data, no GPU)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return {
        k: {"shape": v.get("shape"), "dtype": v.get("dtype")}
        for k, v in header.items()
        if isinstance(v, dict)
    }


def measure_fusion_shapes(model_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Measure fc_embedding / fc_hidden / pre_fc_norm shapes from shard headers.

    Uses ``model.safetensors.index.json`` to find the shards, then reads only
    the safetensors JSON header. This is the required 실측 path.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, encoding="utf-8") as fh:
        weight_map: dict[str, str] = json.load(fh)["weight_map"]
    want = (
        MTP_WEIGHT_NAMES["fc_embedding"],
        MTP_WEIGHT_NAMES["fc_hidden"],
        MTP_WEIGHT_NAMES["pre_fc_norm_embedding"],
        MTP_WEIGHT_NAMES["pre_fc_norm_hidden"],
    )
    shards = {weight_map[k] for k in want if k in weight_map}
    found: dict[str, dict[str, Any]] = {}
    for shard in shards:
        header = read_safetensors_header(model_dir / shard)
        for key in want:
            if key in header:
                found[key] = header[key]
    missing = [k for k in want if k not in found]
    if missing:
        raise FileNotFoundError(f"MTP fusion tensors missing from headers: {missing}")
    return found


def classify_fusion(shapes: Mapping[str, Sequence[int]], hidden_size: int) -> str:
    """Return the fusion recipe name, or raise if the layout is unknown.

    Confirmed recipe (Qwen3-Next/3.5 concat-fc split into two matrices):
      ``fc_embedding`` and ``fc_hidden`` both ``[H, H]`` (PyTorch Linear layout)
      → ``y = linear(norm_e, W_e) + linear(norm_h, W_h)``.

    Not implemented (no official Flash-Next kernel to copy):
      any other pair of shapes, including a leftover ``[H, 2H]`` single-fc layout
      stored under one of the two names.
    """
    we = tuple(shapes[MTP_WEIGHT_NAMES["fc_embedding"]])
    wh = tuple(shapes[MTP_WEIGHT_NAMES["fc_hidden"]])
    h = int(hidden_size)
    if we == (h, h) and wh == (h, h):
        return "split_add"
    raise NotImplementedError(
        "MTP fusion layout is not the split concat-fc ([H,H]+[H,H]). "
        f"Measured fc_embedding={we} fc_hidden={wh} hidden_size={h}. "
        "Official transformers qwen4_exp has no MTP; do not guess a second recipe. "
        "Qwen3-Next/3.5 use a single mtp.fc of [H, 2H] on cat([emb, hidden], -1)."
    )


class _LinearLike(Protocol):
    def __call__(self, x: Any, weight: Any) -> Any: ...


@dataclass
class QwenMtpFusion:
    """Embed/hidden fusion that precedes the MTP decoder block.

    Weights are stored on this object after load; the body supplies ``linear``
    and ``rms_norm`` callables so this file never imports qwen4_exp layers.
    """

    config: QwenMtpConfig
    recipe: str = "unresolved"
    fc_embedding: Any = None
    fc_hidden: Any = None
    pre_fc_norm_embedding: Any = None
    pre_fc_norm_hidden: Any = None

    def resolve_from_measured_shapes(self, measured: Mapping[str, Mapping[str, Any]]) -> str:
        shapes = {k: tuple(v["shape"]) for k, v in measured.items() if "shape" in v}
        self.recipe = classify_fusion(shapes, self.config.hidden_size)
        return self.recipe

    def forward(
        self,
        hidden_states: Any,
        inputs_embeds: Any,
        *,
        linear: _LinearLike,
        rms_norm: Callable[..., Any],
    ) -> Any:
        if self.recipe != "split_add":
            raise NotImplementedError(
                f"fusion recipe={self.recipe!r}. Call resolve_from_measured_shapes "
                "after reading safetensors headers. Official HF qwen4_exp has no MTP."
            )
        if self.config.mtp_use_dedicated_embeddings:
            raise NotImplementedError(
                "mtp_use_dedicated_embeddings=true but this checkpoint ships no "
                "mtp.embed_tokens; dedicated table is unmeasured."
            )
        if any(
            t is None
            for t in (
                self.fc_embedding,
                self.fc_hidden,
                self.pre_fc_norm_embedding,
                self.pre_fc_norm_hidden,
            )
        ):
            raise RuntimeError("QwenMtpFusion weights are not loaded")
        # RMSNorm implementation is injected: body Qwen4ExpTextRMSNorm is 1+weight
        # (zero-init). Re-deriving (1+w) vs w here would be an unconfirmed formula.
        norm_e = rms_norm(inputs_embeds, self.pre_fc_norm_embedding, self.config.rms_norm_eps)
        norm_h = rms_norm(hidden_states, self.pre_fc_norm_hidden, self.config.rms_norm_eps)
        return linear(norm_e, self.fc_embedding) + linear(norm_h, self.fc_hidden)


@dataclass
class QwenMtpDrafter:
    """One-layer MTP drafter. Block internals are injected; this file only sequences them.

    Forward (literature + this checkpoint's tensors, 1-step draft):

    1. ``inputs_embeds = embed_tokens(next_token_ids)``  -- shared table
    2. ``fused = Fusion(hidden_tap, inputs_embeds)``     -- 2560
    3. ``hc = fused.repeat(..., hc_count)``              -- 10240; body TextModel does this
    4. ``hc = decoder_layer_0(hc, ...)``                 -- full_attention + MoE + GR
    5. ``hidden = hyper_connection_mixer(hc)``           -- 2560
    6. ``logits = lm_head(hidden)``                      -- shared; no mtp.norm

    Token offset (Qwen3-Next/vLLM comment, **not** in official qwen4_exp): MTP
    predicts token n+2 from ``(h_n, embed(n+1))``. Unconfirmed for Flash-Next
    "trained with multi-steps" -- see :meth:`draft_offset`.
    """

    config: QwenMtpConfig
    fusion: QwenMtpFusion
    n_target_layers: int | None = None
    # Injected by the body team. None until bind().
    decoder_layer: Any = None
    hyper_connection_mixer: Any = None
    embed_tokens: Callable[[Any], Any] | None = None
    lm_head: Callable[[Any], Any] | None = None
    linear: _LinearLike | None = None
    rms_norm: Callable[..., Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def bind(
        self,
        *,
        n_target_layers: int,
        decoder_layer: Any,
        hyper_connection_mixer: Any,
        embed_tokens: Callable[[Any], Any],
        lm_head: Callable[[Any], Any],
        linear: _LinearLike,
        rms_norm: Callable[..., Any],
    ) -> None:
        if self.config.num_hidden_layers != 1:
            raise NotImplementedError(
                f"checkpoint/config has {self.config.num_hidden_layers} MTP layers; "
                "only the shipped 1-layer stack is scoped."
            )
        if self.config.layer_types != ("full_attention",):
            raise NotImplementedError(
                f"MTP layer_types={self.config.layer_types}; only full_attention is shipped."
            )
        self.n_target_layers = n_target_layers
        self.decoder_layer = decoder_layer
        self.hyper_connection_mixer = hyper_connection_mixer
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.linear = linear
        self.rms_norm = rms_norm

    @property
    def extra_moe_layers(self) -> int:
        """Value to add to ``ModelConfig.extra_moe_layers`` when MTP is enabled."""
        return self.config.num_hidden_layers

    @property
    def draft_moe_layer_id(self) -> int:
        if self.n_target_layers is None:
            raise RuntimeError("bind() first")
        return self.n_target_layers  # first (only) MTP block

    def draft_offset(self) -> int:
        """How many tokens ahead of the hidden tap the first MTP logit predicts.

        Qwen3-Next / Qwen3.5 ports: 1 (hidden at n, embed of token n+1 → token n+2).
        Official qwen4_exp: no MTP, so this offset is unconfirmed for Flash-Next.
        """
        raise NotImplementedError(
            "Flash-Next MTP token offset is not in official transformers qwen4_exp. "
            "Qwen3-Next/vLLM: predict n+2 from (h_n, embed(n+1)). "
            "README says 'MTP: 1 layer, trained with multi-steps' -- multi-step "
            "recurrence reuses this offset, but the first-step pairing is unmeasured."
        )

    def expand_hc(self, fused: Any) -> Any:
        """Repeat last dim by hc_count. Matches Qwen4ExpTextModel (official)."""
        # fused is [..., H]; body does hidden.repeat(1, 1, hc_count) → [..., H*hc_count].
        repeat = getattr(fused, "repeat", None)
        if repeat is None:
            raise TypeError("expand_hc expects a tensor with .repeat")
        return repeat(1, 1, self.config.hc_count)

    def forward(
        self,
        hidden_states: Any,
        next_token_ids: Any,
        *,
        inputs_embeds: Any | None = None,
        decoder_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        if self.config.mtp_use_hidden_state_from_layer is not None:
            raise NotImplementedError(
                "mtp_use_hidden_state_from_layer="
                f"{self.config.mtp_use_hidden_state_from_layer}: mid-stack tap "
                "is not implemented (null on this checkpoint means last mixer output)."
            )
        if any(
            x is None
            for x in (
                self.embed_tokens,
                self.lm_head,
                self.decoder_layer,
                self.hyper_connection_mixer,
                self.linear,
                self.rms_norm,
            )
        ):
            raise RuntimeError("QwenMtpDrafter.bind() was not called")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(next_token_ids)
        fused = self.fusion.forward(
            hidden_states,
            inputs_embeds,
            linear=self.linear,
            rms_norm=self.rms_norm,
        )
        hc = self.expand_hc(fused)
        hc = self.decoder_layer(hc, **(decoder_kwargs or {}))
        hidden = self.hyper_connection_mixer(hc)
        return self.lm_head(hidden)

    def catch_up_draft_context(self, *args: Any, **kwargs: Any) -> None:
        """Fill MTP GQA/QSA cache for tokens the target just committed.

        DSV4 dSpark does this by projecting target hidden through each draft
        layer's ``wkv`` (MLA) without running the prompt. MTP has no MLA ``wkv``
        -- it has GQA ``k_proj``/``v_proj`` plus a QSA indexer. Whether writing
        K/V from the *target* hidden (untrained for that mapping) or replaying
        the MTP block over committed tokens is correct is unconfirmed.
        """
        raise NotImplementedError(
            "MTP catch-up is not the DSV4 wkv projection. Options (unconfirmed): "
            "(1) run the MTP block over committed tokens with the target hidden "
            "tap -- standard Qwen3.5/vLLM prefill; "
            "(2) write GQA K/V from k_proj/v_proj(target_hidden) -- cheaper, "
            "probably off-distribution. Official qwen4_exp has no MTP cache API."
        )

    def draft(self, *args: Any, **kwargs: Any) -> Any:
        """One-token (or multi-step recurrent) proposal. Engine wires this later."""
        raise NotImplementedError(
            "draft() waits on body decoder + engine speculative loop. "
            "Unlike DSpark.propose this is NOT a block sampler (no markov/confidence "
            "heads in the checkpoint). Recurrent multi-step needs draft_offset()."
        )


def expected_expert_fp8_bytes(cfg: QwenMtpConfig) -> int:
    """Routed-expert FP8 payload: 512 × 3 × H × I bytes (e4m3, no scales)."""
    return (
        cfg.num_experts
        * 3
        * cfg.hidden_size
        * cfg.moe_intermediate_size
    )


def load_hf_config(model_dir: str | Path) -> dict[str, Any]:
    with open(Path(model_dir) / "config.json", encoding="utf-8") as fh:
        return json.load(fh)


def build_unbound_drafter(hf_config: Mapping[str, Any] | Any) -> QwenMtpDrafter:
    cfg = parse_mtp_config(hf_config)
    return QwenMtpDrafter(config=cfg, fusion=QwenMtpFusion(config=cfg))


def iter_mtp_index_keys(weight_map: Mapping[str, str]) -> Iterable[str]:
    for name in weight_map:
        if name.startswith(CKPT_PREFIX):
            yield name


__all__ = [
    "BODY_BINDING_POINTS",
    "CKPT_PREFIX",
    "MTP_LAYER0_PREFIXES",
    "MTP_WEIGHT_NAMES",
    "QwenMtpConfig",
    "QwenMtpDrafter",
    "QwenMtpFusion",
    "build_unbound_drafter",
    "classify_fusion",
    "expected_expert_fp8_bytes",
    "iter_mtp_index_keys",
    "load_hf_config",
    "measure_fusion_shapes",
    "mtp_expert_weight_names",
    "parse_mtp_config",
    "read_safetensors_header",
]
