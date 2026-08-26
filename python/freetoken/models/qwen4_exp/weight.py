from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import iter_weight_files
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

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
#   expert_bank  -> routed experts (fp8 + scales), loaded by setup_offload_expert_banks
#   ngram_module -> ple.ple_embedding.* subtree, mmap-loaded directly by ngram.py
#   dropped      -> mtp.* / vision / quant sidecar scales we intentionally do not serve
# ---------------------------------------------------------------------------

_EXPERT_RE = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.")
_DROP_PREFIXES = ("mtp.", "model.visual.", "visual.")
_DROP_SUFFIXES = (".k_scale", ".v_scale", ".q_scale", ".prob_scale")

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
    """-> (category, detail). detail = renamed dense key, or the reason for the rest.
    Categories: dense / expert_bank / ngram_module / dropped / unknown."""
    if raw_name.startswith(_DROP_PREFIXES):
        return "dropped", "mtp/vision not served in P0"
    if raw_name.endswith(_DROP_SUFFIXES):
        return "dropped", "quant sidecar scale (engine keeps native-precision KV)"
    if _EXPERT_RE.match(raw_name):
        return "expert_bank", "routed expert (fp8_block offload banks)"
    if ".ple.ple_embedding." in raw_name:
        return "ngram_module", "n-gram table/buffers, mmap-loaded by ngram.py"
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
