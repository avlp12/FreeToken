"""Qwen3.8-Flash-Next / qwen4_exp checkpoint layout helpers (stdlib only).

Used by the FTW converter and ``ftw_qwen_dryrun`` to classify tensors without
importing ``models/qwen4_exp`` (that package is owned by another worker).
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from typing import Any, Iterable, Iterator

# Destination kinds the converter writes (or drops).
DEST_SKIP = "skip"
DEST_EXPERT = "expert"
DEST_NGRAM = "ngram"
DEST_DENSE = "dense"

# Explicit skip: MTP draft head + vision tower. Text-only FTW must not carry these;
# ``iter_ftw_weights`` only drops VISION_KEY_PREFIXES (vision_tower./embed_vision.),
# which do NOT match Qwen's ``model.visual.*``.
SKIP_PREFIXES = ("mtp.", "model.visual.", "visual.")

# Unused ModelOpt KV-cache static scales (same drop as qwen3_5_moe._rename).
_SKIP_SUFFIXES = (".k_scale", ".v_scale", ".q_scale", ".prob_scale")

# Routed experts on the *language* tower only. The ``language_model.layers`` (or
# post-rename ``model.layers``) anchor excludes ``mtp.layers.*.mlp.experts.*``.
_EXPERT_RE = re.compile(
    r"(?:^model\.language_model\.|^language_model\.|^model\.)"
    r"layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate|up|down)_proj\.(?:weight|weight_scale_inv)$"
)

# N-gram / PLE embedding table. Observed on Qwen3.8-Flash-Next-FP8:
#   model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{N}.weight
#   (+ optional sibling weight_scale_inv). Small PLE projections stay dense.
_NGRAM_RE = re.compile(
    r"(ngram_embedding|embed_ngram|ngram_embed|ngram_table|split_ngram|"
    r"ple\.(?:embed|table|ngram|weight_part)|"
    r"ple_embedding\.ngram)",
    re.I,
)
_PLE_SMALL_RE = re.compile(
    r"\.ple\.(?:conv1d|key_proj|value_proj|norm|in_proj|out_proj)\b",
    re.I,
)

_NGRAM_SHARD_RE = re.compile(
    r"(?:^|[^a-z0-9])ngram(?:[^a-z0-9]|$)|ngram_emb|ngram-embed",
    re.I,
)

# safetensors header dtype -> element size (dry-run size math; no torch).
_ST_ELSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}

ALIGN = 4096  # must match checkpoint.ftw.ALIGN
DEFAULT_SHARD_LIMIT = 8 << 30
QWEN4_ARCH_PREFIXES = ("Qwen4Exp",)
QWEN4_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_hf_config_dict(model_path: str) -> dict:
    return load_json(os.path.join(model_path, "config.json"))


def is_wrapper_config(cfg: dict) -> bool:
    """True when HF config is a multimodal wrapper with a nested text tower."""
    text = cfg.get("text_config")
    return isinstance(text, dict) and bool(text)


def unwrap_text_config(cfg: dict) -> dict:
    """Return the text-tower config, falling back to the top-level dict."""
    text = cfg.get("text_config")
    return text if isinstance(text, dict) else cfg


def is_qwen4_exp_config(cfg: dict) -> bool:
    arch = cfg.get("architectures") or []
    if any(isinstance(a, str) and a.startswith(QWEN4_ARCH_PREFIXES) for a in arch):
        return True
    mt = str(cfg.get("model_type") or "")
    if mt in QWEN4_MODEL_TYPES:
        return True
    text = unwrap_text_config(cfg)
    return str(text.get("model_type") or "") in QWEN4_MODEL_TYPES


def classify_tensor(name: str) -> str:
    """Map a raw HF weight-map key to an FTW destination."""
    if name.startswith(SKIP_PREFIXES) or name.endswith(_SKIP_SUFFIXES):
        return DEST_SKIP
    if _EXPERT_RE.search(name):
        return DEST_EXPERT
    if _PLE_SMALL_RE.search(name):
        return DEST_DENSE
    if _NGRAM_RE.search(name):
        return DEST_NGRAM
    return DEST_DENSE


def load_weight_map(model_path: str) -> dict[str, str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        return load_json(index_path)["weight_map"]
    weight_map: dict[str, str] = {}
    for shard in sorted(os.path.basename(p) for p in glob.glob(os.path.join(model_path, "*.safetensors"))):
        hdr = read_safetensors_header(os.path.join(model_path, shard))
        for nm in hdr:
            if nm != "__metadata__":
                weight_map[nm] = shard
    return weight_map


def extra_safetensor_files(model_path: str, weight_map: dict[str, str]) -> list[str]:
    """Safetensors in ``model_path`` that the main index does not mention.

    N-gram tables are sometimes shipped as a parallel 128-shard set (or a second
    index). Those files must still land in the FTW.
    """
    indexed = set(weight_map.values())
    extra = []
    for path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        name = os.path.basename(path)
        if name not in indexed:
            extra.append(name)
    return extra


def looks_like_ngram_shard(filename: str) -> bool:
    return _NGRAM_SHARD_RE.search(filename) is not None


def extra_index_files(model_path: str) -> list[str]:
    skip = {"model.safetensors.index.json"}
    found = []
    for path in sorted(glob.glob(os.path.join(model_path, "*index.json"))):
        name = os.path.basename(path)
        if name not in skip:
            found.append(name)
    return found


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n <= 0 or n > 256 << 20:
            raise ValueError(f"implausible safetensors header size {n} in {path}")
        return json.loads(fh.read(n))


def tensor_nbytes_from_meta(meta: dict) -> int:
    if "data_offsets" in meta:
        b, e = meta["data_offsets"]
        return int(e) - int(b)
    shape = meta.get("shape") or []
    dt = meta.get("dtype") or "BF16"
    el = _ST_ELSIZE.get(dt)
    if el is None:
        raise ValueError(f"unknown safetensors dtype {dt!r}")
    n = 1
    for d in shape:
        n *= int(d)
    return n * el


def iter_shard_tensor_metas(model_path: str, shards: Iterable[str]) -> Iterator[tuple[str, str, dict, int]]:
    """Yield ``(name, shard, meta, nbytes)`` from safetensors headers (no weight bytes)."""
    for shard in shards:
        path = os.path.join(model_path, shard)
        hdr = read_safetensors_header(path)
        for name, meta in hdr.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            yield name, shard, meta, tensor_nbytes_from_meta(meta)


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def ftw_padded_bytes(nbytes: int) -> int:
    """On-disk FTW occupancy of one tensor (payload + 4096 pad)."""
    return align_up(nbytes)


def estimate_fp8_block_banks(
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate: int,
    block: int = 128,
) -> dict[str, int]:
    """Per-bank and total payload bytes for qwen3_5_moe-style fp8_block expert banks.

    Layout (one ``[E, ...]`` HostBank per layer, streamed as ``{name}#L{layer:05d}``):

    * gate_up        [E, 2I, H]     fp8
    * gate_up_scale  [E, 2I/B, H/B] bf16
    * down           [E, H, I]      fp8
    * down_scale     [E, H/B, I/B]  bf16
    """
    e, h, i, b = num_experts, hidden_size, moe_intermediate, block
    assert (2 * i) % b == 0 and h % b == 0 and i % b == 0, (h, i, b)
    per_layer = {
        "gate_up": e * (2 * i) * h,
        "gate_up_scale": e * (2 * i // b) * (h // b) * 2,
        "down": e * h * i,
        "down_scale": e * (h // b) * (i // b) * 2,
    }
    layers = {
        name: num_layers * nbytes for name, nbytes in per_layer.items()
    }
    return {
        "per_layer": per_layer,
        "per_bank": layers,
        "payload": sum(layers.values()),
        "n_entries": num_layers * len(per_layer),
        "padded": sum(num_layers * ftw_padded_bytes(n) for n in per_layer.values()),
    }


def estimate_ngram_table(text: dict) -> dict[str, int] | None:
    """Config-only n-gram payload (used when keys are not yet visible in the index)."""
    vocab = int(text.get("ngram_vocab_size_base") or 0)
    hidden = int(text.get("ple_embed_dim") or text.get("hidden_size") or 0)
    parts = int(text.get("split_ngram_parts") or 0)
    if vocab <= 0 or hidden <= 0:
        return None
    # fp8 embedding + 128x128 bf16 block scale (same scheme as the rest of the ckpt).
    weight = vocab * hidden
    bn, bk = 128, 128
    rows = (vocab + bn - 1) // bn
    cols = (hidden + bk - 1) // bk
    scale = rows * cols * 2
    return {
        "vocab": vocab,
        "hidden": hidden,
        "parts": parts,
        "weight": weight,
        "scale": scale,
        "payload": weight + scale,
    }


def disk_free_bytes(path: str) -> int:
    """Filesystem free bytes for ``path`` (creates nothing)."""
    probe = path if os.path.exists(path) else os.path.dirname(os.path.abspath(path)) or os.sep
    if hasattr(os, "statvfs"):
        st = os.statvfs(probe)
        return int(st.f_bavail) * int(st.f_frsize)
    import shutil

    return int(shutil.disk_usage(probe).free)


__all__ = [
    "ALIGN",
    "DEFAULT_SHARD_LIMIT",
    "DEST_DENSE",
    "DEST_EXPERT",
    "DEST_NGRAM",
    "DEST_SKIP",
    "SKIP_PREFIXES",
    "classify_tensor",
    "disk_free_bytes",
    "estimate_fp8_block_banks",
    "estimate_ngram_table",
    "extra_index_files",
    "extra_safetensor_files",
    "ftw_padded_bytes",
    "is_qwen4_exp_config",
    "is_wrapper_config",
    "iter_shard_tensor_metas",
    "load_hf_config_dict",
    "load_json",
    "load_weight_map",
    "looks_like_ngram_shard",
    "read_safetensors_header",
    "tensor_nbytes_from_meta",
    "unwrap_text_config",
]
