from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import vision_load_enabled
from freetoken.models.loader import iter_weight_files
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .vision import adapt_vision_tensor

# The routed-expert bank machinery is IDENTICAL to qwen3_5-FP8 (same block-fp8 layout,
# same ``model.language_model.layers.N.mlp.experts.E.{gate|up|down}_proj`` keys, verified
# against the shipped index.json), so the bank builder is reused wholesale.
from freetoken.models.qwen3_5_moe.weight import (  # noqa: F401  (re-exported hooks)
    setup_offload_expert_banks,
)

from .config import parse_config  # noqa: F401  (spec contract)

# ---------------------------------------------------------------------------
# Key classification -- the single source of truth shared by iter_weights and
# tools/qwen4_map_check.py. Categories:
#   dense        -> streamed through iter_weights (renamed, maybe fused / +1-baked)
#   vision       -> model.visual.* (renamed vision_tower.*), streamed through iter_weights
#                   IFF vision_load_enabled() (opt-in; the model only builds a
#                   vision_tower submodule under the same flag, see qwen4_exp/model.py).
#                   checkpoint/convert.py's FTW pass-through carries these tensors
#                   unconditionally regardless of this flag -- see its docstring for why
#                   conversion-time and load-time gating are deliberately different.
#   expert_bank  -> routed experts (fp8 + scales), loaded by setup_offload_expert_banks
#   ngram_module -> ple.ple_embedding.* subtree, mmap-loaded directly by ngram.py
#   mtp_dense / mtp_expert_bank -> MTP drafter head, preserved for a later phase
#   dropped      -> quant sidecar scales we intentionally do not serve
# ---------------------------------------------------------------------------

_EXPERT_RE = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.")
# model.visual.* / visual.* (vision tower) -- classified below as its own "vision"
# category (NOT dropped: see the class comment above and adapt_vision_tensor's docstring).
_VISION_PREFIXES = ("model.visual.", "visual.")
# mtp.* is classified below (mtp_dense / mtp_expert_bank), NOT dropped: the MTP drafter
# head is preserved for a later phase (see MTP_DENSE_PATTERNS / MTP_EXPERT_RE).
_DROP_SUFFIXES = (".k_scale", ".v_scale", ".q_scale", ".prob_scale")

# MTP drafter routed-expert bank: 512 experts x {gate,up,down}_proj x {weight,
# weight_scale_inv} = 3072 tensors, fp8 block-scaled, same layout as the target's own
# routed experts but under the mtp. prefix (own bank, own layer). Preserved in the FTW
# (kind="mtp_expert", see checkpoint/convert.py) but NOT streamed here -- like the
# target's expert_bank category, it is not a load_state_dict dense parameter.
_MTP_EXPERT_RE = re.compile(r"^mtp\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")

# MTP drafter dense (bf16) tensors: fc_embedding/fc_hidden + norms, the mixer, and one
# full decoder layer (self_attn + indexer, MoE gate/shared_expert, two hyper_connections).
# 29 tensors total (verified against tools/qwen4_map_check.py). Preserved in the FTW
# (kind="mtp_dense") but NOT streamed here -- no Qwen4ExpForCausalLM.mtp submodule
# exists yet to receive them via load_state_dict; wiring them up is a later phase.
_MTP_DENSE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"^mtp\.(fc_embedding|fc_hidden|pre_fc_norm_embedding|pre_fc_norm_hidden)\.weight$",
        r"^mtp\.hyper_connection_mixer\.(hc_norm|input_mix_weight_down|input_mix_weight_up)\.weight$",
        r"^mtp\.layers\.0\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight$",
        r"^mtp\.layers\.0\.self_attn\.indexer\.(index_qk_proj|q_layernorm|k_layernorm)\.weight$",
        r"^mtp\.layers\.0\.mlp\.(gate|shared_expert_gate)\.weight$",
        r"^mtp\.layers\.0\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$",
        r"^mtp\.layers\.0\.(attn_hyper_connection|mlp_hyper_connection)\."
        r"(hc_norm|input_mix_weight_down|input_mix_weight_up|block_inject_weight)\.weight$",
    )
)

# Gemma-style (1+weight) RMSNorms: every Qwen4ExpTextRMSNorm instance (zeros-init,
# forward computes normed * (1 + w)); the loader bakes the +1 so runtime modules do a
# raw multiply. The GDN gated norm (linear_attn.norm) is ones-init raw -- excluded.
_GEMMA_NORM_SUFFIXES = (
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".hc_norm.weight",
    ".indexer.q_layernorm.weight",
    ".indexer.k_layernorm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
)

# Fused projections: concat checkpoint matrices in this order to match the modules'
# LinearColParallelMerged splits.
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ),
}


def _rename(raw_name: str) -> str:
    name = raw_name
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model.") :]
    return name


def _rename_vision(raw_name: str) -> str:
    """``model.visual.<rest>`` / ``visual.<rest>`` -> ``vision_tower.<rest>`` -- the prefix
    ``Qwen4ExpForCausalLM``'s ``vision_tower`` submodule puts on its own state-dict keys.
    Same rule as ``checkpoint/qwen_layout.py``'s ``ftw_vision_name`` (kept as an independent
    implementation there since that module is deliberately stdlib-only / models-free)."""
    for prefix in _VISION_PREFIXES:
        if raw_name.startswith(prefix):
            return "vision_tower." + raw_name[len(prefix):]
    raise ValueError(f"not a vision tensor name: {raw_name!r}")


# Exhaustive whitelist of expected DENSE keys (raw checkpoint form). Anything that
# matches no category below is UNKNOWN -- the map-check tool fails and iter_weights
# raises, instead of silently feeding load_state_dict a surprise.
_LAYER = r"^model\.language_model\.layers\.\d+\."
_DENSE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"^lm_head\.weight$",
        r"^model\.language_model\.embed_tokens\.weight$",
        r"^model\.language_model\.hyper_connection_mixer\."
        r"(hc_norm|input_mix_weight_down|input_mix_weight_up)\.weight$",
        _LAYER + r"(attn|mlp)_hyper_connection\."
        r"(hc_norm|input_mix_weight_down|input_mix_weight_up|block_inject_weight)\.weight$",
        _LAYER + r"linear_attn\.(A_log|dt_bias)$",
        _LAYER + r"linear_attn\."
        r"(conv1d|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|norm|out_proj)\.weight$",
        _LAYER + r"mlp\.(gate|shared_expert_gate)\.weight$",
        _LAYER + r"mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$",
        _LAYER + r"self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight$",
        _LAYER + r"self_attn\.indexer\.(index_qk_proj|q_layernorm|k_layernorm)\.weight$",
        _LAYER + r"ple\.(conv1d|key_proj|value_proj|norm_conv|norm_key|norm_query)\.weight$",
    )
)


def classify_key(raw_name: str) -> tuple[str, str]:
    """-> (category, detail). detail = renamed dense/vision key, or the reason for the rest.
    Categories: dense / vision / expert_bank / ngram_module / mtp_dense / mtp_expert_bank /
    dropped / unknown."""
    if raw_name.startswith(_VISION_PREFIXES):
        return "vision", _rename_vision(raw_name)
    if raw_name.endswith(_DROP_SUFFIXES):
        return "dropped", "quant sidecar scale (engine keeps native-precision KV)"
    if _MTP_EXPERT_RE.match(raw_name):
        return "mtp_expert_bank", "MTP drafter routed expert (fp8, preserved for a later phase)"
    if _EXPERT_RE.match(raw_name):
        return "expert_bank", "routed expert (fp8_block offload banks)"
    if ".ple.ple_embedding." in raw_name:
        return "ngram_module", "n-gram table/buffers, mmap-loaded by ngram.py"
    if any(p.match(raw_name) for p in _MTP_DENSE_PATTERNS):
        return "mtp_dense", "MTP drafter dense weight (preserved for a later phase)"
    if any(p.match(raw_name) for p in _DENSE_PATTERNS):
        return "dense", _rename(raw_name)
    return "unknown", raw_name


def _is_gemma_norm(name: str) -> bool:
    return name.endswith(_GEMMA_NORM_SUFFIXES)


def _try_fuse(name, tensor, buf):
    for fused_suffix, parts in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if name.endswith(part):
                key = name[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return key, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream the DENSE weights (everything bf16 in the FP8 release). Routed experts are
    ALWAYS excluded here -- they only exist as fp8_block offload banks
    (setup_offload_expert_banks); a resident-MoE run would fail loudly on missing expert
    keys in load_state_dict, which is intended (240 GB dequantized does not fit)."""
    _ = cached_load_hf_config(model_path)  # parity with qwen3_5 (validates the config)
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading currently supports TP=1 only")
    if not include_non_moe:
        return  # experts never stream through here (offload banks only)

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                category, detail = classify_key(raw_name)
                if category == "unknown":
                    raise ValueError(f"unexpected checkpoint key: {raw_name}")
                if category == "vision":
                    # Opt-in, matching qwen4_exp/model.py only building a vision_tower
                    # submodule under the same flag -- yielding these unconditionally
                    # would make load_state_dict see unexpected keys when vision is off.
                    # checkpoint/convert.py's FTW pass-through carries vision tensors
                    # unconditionally regardless of this flag (see its docstring); this
                    # gate only applies to this raw-HF-checkpoint streaming path.
                    if not vision_load_enabled():
                        continue
                    name = detail  # "vision_tower.<short>"
                    short = name[len("vision_tower."):]
                    tensor = adapt_vision_tensor(short, f.get_tensor(raw_name))
                    yield name, tensor
                    continue
                if category != "dense":
                    continue
                name = detail
                tensor = f.get_tensor(raw_name)
                if _is_gemma_norm(name):
                    # runtime norms do a raw multiply; the checkpoint stores w for (1+w)
                    tensor = tensor.float().add_(1.0).to(tensor.dtype)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is None:
                    yield name, tensor
                elif fused:
                    yield fused

    if fuse_buf:
        missing = {k: sorted(v.keys()) for k, v in fuse_buf.items()}
        raise RuntimeError(f"incomplete fusion groups at end of checkpoint: {missing}")


__all__ = ["iter_weights", "setup_offload_expert_banks", "classify_key", "parse_config"]
