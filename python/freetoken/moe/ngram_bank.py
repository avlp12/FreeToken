"""Host-resident n-gram embedding bank (Qwen4-Exp / Qwen3.8-Flash-Next PLE).

Interface contract
==================
This module is the high-performance stand-in for the naive in-process table in
``freetoken.models.qwen4_exp.ngram``. A caller that already turned token
n-grams into **global padded-table row ids** can swap implementations by
constructing a bank and calling ``lookup`` / ``lookup_to``. This bank does
**not** hash, apply ``layer_multipliers``, or add ``ngram_heads_offsets`` —
those stay in the PLE module.

Global row-id meaning
---------------------
``global_ids[..., h]`` is a row index into the **concatenated** n-gram table:
shard_0 rows, then shard_1, …, then shard_127, in that numeric order. The
padded vocab is ``sum(shard_i.n_rows)``. Per-head hash ids must already be
``(hash % ngram_heads_vocab_sizes[h]) + ngram_heads_offsets[h]`` before they
reach this bank.

Public API
----------
- ``NGramBank.open(model_path)`` — parse ``model.safetensors.index.json``,
  mmap every safetensors file that holds an n-gram shard (header-only parse;
  **no** full-file or full-table materialization). Builds a per-shard
  ``(file view, abs_offset, shape)`` table. Tiny metadata tensors
  (``weight_scale``, ``ngram_heads_offsets``, ``ngram_heads_vocab_sizes``)
  are copied into RAM.
- ``lookup(global_ids)`` — ``LongTensor[..., H] -> bfloat16 CPU Tensor[..., H, D]``.
  For Flash-Next: ``H=16``, ``D=160``. Converts global ids to
  ``(shard, local_row)``, gathers fp8 rows, dequants with the per-tensor
  ``weight_scale``: ``bf16 = fp8.to(bf16) * scale``.
- ``lookup_to(global_ids, device, non_blocking=True)`` — ``lookup`` then
  optional pinned-host staging + async H2D. Decode traffic is
  ``16 * 160 * 1 B ≈ 2.5 KB`` of fp8 (5 KB after bf16); pin+copy latency is
  in the noise next to the rest of a decode step.
- ``gather_fp8(global_ids)`` — same gather **without** dequant (fp8 codes).
  Used to prove mmap bits match the safetensors library path.

Storage (measured on Qwen3.8-Flash-Next-FP8)
--------------------------------------------
- 128 shards, each ``F8_E4M3`` ``[2500012, 160]`` (~400 MiB).
- ``weight_scale``: ``BF16`` ``[1]`` (one scalar for the whole table).
- ``ngram_heads_offsets`` / ``ngram_heads_vocab_sizes``: ``I64`` ``[16]``.
- Padded vocab = ``320001536``. Files are ≤ ~1.72 GiB; only the header JSON
  is parsed. Data stays file-backed.

Design (mmap vs pinned; where dequant runs)
-------------------------------------------
A pinned 51 GiB host bank would lock the whole table in RAM and need
``cudaHostAlloc``. mmap (``ACCESS_READ``) publishes read-only file mappings
and lets the OS page cache keep hot rows; cold shards cost virtual address
space only. Dequant is on CPU after gather: the scale is a 2-byte RAM
scalar, 16×160 fp8→bf16 is microseconds, and ``lookup_to`` then ships the
already-dequantized bf16 tile. Rejected: GPU-side dequant (kernel launch
dwarfs 2.5 KB) and safetensors ``get_tensor`` on the hot path (it materializes
a 400 MiB shard).
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# safetensors header dtype strings. Duplicated (not imported from
# ``models.weight``) so this module stays a leaf the ngram worker can swap in.
_ST_DTYPE: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

_SHARD_KEY_MARK = ".ngram_embedding.shard_"
_SHARD_KEY_SUFFIX = ".weight"
_INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class _ShardRef:
    """One n-gram shard: mmap-backed fp8 view plus its global-row range."""

    table: torch.Tensor  # [n_rows, head_dim] float8, file-backed
    n_rows: int
    global_start: int  # inclusive
    global_end: int  # exclusive


def _read_st_header(mm: mmap.mmap) -> tuple[int, dict[str, Any]]:
    n = struct.unpack_from("<Q", mm, 0)[0]
    hdr = json.loads(bytes(mm[8 : 8 + n]))
    return 8 + n, hdr


def _map_file(path: str) -> tuple[Any, mmap.mmap, torch.Tensor]:
    """Read-only mmap of ``path`` as a uint8 torch view. Does not copy bytes.

    ``torch.frombuffer`` on a read-only mmap is the common path. The file
    handle and mmap are returned so the view's backing store outlives the
    tensor (frombuffer keeps a buffer ref, but we pin the OS mapping too).
    """
    fh = open(path, "rb")
    size = os.path.getsize(path)
    mm = mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ)
    try:
        # Read-only mmap: we never write through the tensor. Torch warns because
        # the storage is technically mutable via the tensor API; ignore that.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            u8 = torch.frombuffer(mm, dtype=torch.uint8)
    except (ValueError, RuntimeError, TypeError):
        # Some torch builds require a writable buffer; numpy memmap is
        # read-only and still shares pages (no 1.7 GiB copy).
        import numpy as np

        arr = np.memmap(path, dtype=np.uint8, mode="r")
        u8 = torch.from_numpy(arr)
        return (fh, mm, arr), mm, u8
    return (fh, mm), mm, u8


def _tensor_from_u8(
    file_u8: torch.Tensor,
    data_base: int,
    meta: dict[str, Any],
) -> torch.Tensor:
    start, end = meta["data_offsets"]
    raw = file_u8[data_base + start : data_base + end]
    dtype = _ST_DTYPE[meta["dtype"]]
    viewed = raw.view(dtype)
    shape = meta["shape"]
    return viewed if not shape else viewed.view(*shape)


def _device_is_cpu(device: torch.device | str | int | None) -> bool:
    if device is None:
        return True
    if isinstance(device, int):
        return False
    if isinstance(device, str):
        return device == "cpu" or device.startswith("cpu:")
    return device.type == "cpu"


class NGramBank:
    """Mmap-backed fp8 n-gram embedding table with CPU gather + bf16 dequant."""

    def __init__(self) -> None:
        self.model_path: str = ""
        self.prefix: str = ""
        self.num_shards: int = 0
        self.head_dim: int = 0
        self.n_heads: int = 0
        self.padded_vocab: int = 0
        self.weight_scale: torch.Tensor = torch.ones(1, dtype=torch.bfloat16)
        self.ngram_heads_offsets: torch.Tensor = torch.empty(0, dtype=torch.int64)
        self.ngram_heads_vocab_sizes: torch.Tensor = torch.empty(0, dtype=torch.int64)
        self._shards: list[_ShardRef] = []
        self._row_starts: torch.Tensor = torch.empty(0, dtype=torch.int64)
        self._row_ends: torch.Tensor = torch.empty(0, dtype=torch.int64)
        self._keep: list[Any] = []
        self._closed: bool = True

    @classmethod
    def open(cls, model_path: str) -> NGramBank:
        """Map a checkpoint's n-gram shards. ``model_path`` is the HF folder."""
        bank = cls()
        bank._open(os.path.abspath(model_path))
        return bank

    def _open(self, model_path: str) -> None:
        index_path = os.path.join(model_path, _INDEX_NAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"missing {_INDEX_NAME} under {model_path}")
        with open(index_path, encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        shard_keys = self._discover_shard_keys(weight_map)
        prefix = shard_keys[0][: shard_keys[0].index(_SHARD_KEY_MARK) + len(".ngram_embedding")]
        self.model_path = model_path
        self.prefix = prefix

        needed: dict[str, list[str]] = {}
        for name in shard_keys:
            needed.setdefault(weight_map[name], []).append(name)
        for extra in (
            f"{prefix}.weight_scale",
            prefix.replace(".ngram_embedding", ".ngram_heads_offsets"),
            prefix.replace(".ngram_embedding", ".ngram_heads_vocab_sizes"),
        ):
            if extra in weight_map:
                needed.setdefault(weight_map[extra], []).append(extra)

        file_views: dict[str, tuple[int, dict[str, Any], torch.Tensor]] = {}
        for shard_file, names in needed.items():
            path = os.path.join(model_path, shard_file)
            keep, mm, file_u8 = _map_file(path)
            self._keep.append(keep)
            data_base, hdr = _read_st_header(mm)
            file_views[shard_file] = (data_base, hdr, file_u8)
            for name in names:
                if name not in hdr:
                    raise KeyError(f"{name} missing from safetensors header of {shard_file}")

        parsed: list[tuple[int, torch.Tensor]] = []
        for name in shard_keys:
            idx = self._shard_index(name)
            shard_file = weight_map[name]
            data_base, hdr, file_u8 = file_views[shard_file]
            table = _tensor_from_u8(file_u8, data_base, hdr[name])
            if table.ndim != 2:
                raise ValueError(f"{name}: expected rank-2 shard, got {tuple(table.shape)}")
            parsed.append((idx, table))
        parsed.sort(key=lambda item: item[0])
        if [i for i, _ in parsed] != list(range(len(parsed))):
            raise ValueError(f"shard indices are not contiguous 0..{len(parsed) - 1}")

        shards: list[_ShardRef] = []
        cursor = 0
        head_dim = parsed[0][1].shape[1]
        for idx, table in parsed:
            if table.shape[1] != head_dim:
                raise ValueError(f"shard {idx} head_dim {table.shape[1]} != {head_dim}")
            n_rows = int(table.shape[0])
            shards.append(_ShardRef(table=table, n_rows=n_rows, global_start=cursor, global_end=cursor + n_rows))
            cursor += n_rows

        self._shards = shards
        self.num_shards = len(shards)
        self.head_dim = int(head_dim)
        self.padded_vocab = cursor
        # Invariant: global id i lives in shard s iff row_starts[s] <= i < row_ends[s].
        # searchsorted(row_ends, i, right=True) is that s. Tables are immutable
        # mmap views published here, before any lookup.
        self._row_starts = torch.tensor([s.global_start for s in shards], dtype=torch.int64)
        self._row_ends = torch.tensor([s.global_end for s in shards], dtype=torch.int64)

        scale_key = f"{prefix}.weight_scale"
        if scale_key in weight_map:
            data_base, hdr, file_u8 = file_views[weight_map[scale_key]]
            self.weight_scale = _tensor_from_u8(file_u8, data_base, hdr[scale_key]).detach().clone()
        else:
            self.weight_scale = torch.ones(1, dtype=torch.bfloat16)

        off_key = prefix.replace(".ngram_embedding", ".ngram_heads_offsets")
        vs_key = prefix.replace(".ngram_embedding", ".ngram_heads_vocab_sizes")
        if off_key in weight_map:
            data_base, hdr, file_u8 = file_views[weight_map[off_key]]
            self.ngram_heads_offsets = _tensor_from_u8(file_u8, data_base, hdr[off_key]).detach().clone().to(torch.int64)
        if vs_key in weight_map:
            data_base, hdr, file_u8 = file_views[weight_map[vs_key]]
            self.ngram_heads_vocab_sizes = (
                _tensor_from_u8(file_u8, data_base, hdr[vs_key]).detach().clone().to(torch.int64)
            )
        self.n_heads = int(self.ngram_heads_offsets.numel()) if self.ngram_heads_offsets.numel() else 16
        self._closed = False
        logger.info(
            "NGramBank open %s shards=%d rows=%d head_dim=%d scale=%s dtype=%s",
            model_path,
            self.num_shards,
            self.padded_vocab,
            self.head_dim,
            tuple(self.weight_scale.shape),
            shards[0].table.dtype,
        )

    @staticmethod
    def _discover_shard_keys(weight_map: dict[str, str]) -> list[str]:
        found: dict[str, list[str]] = {}
        for name in weight_map:
            if _SHARD_KEY_MARK not in name or not name.endswith(_SHARD_KEY_SUFFIX):
                continue
            prefix = name[: name.index(_SHARD_KEY_MARK) + len(".ngram_embedding")]
            found.setdefault(prefix, []).append(name)
        if not found:
            raise KeyError("no ngram_embedding.shard_*.weight keys in index.json")
        prefix = sorted(found)[0]
        keys = found[prefix]
        keys.sort(key=NGramBank._shard_index)
        return keys

    @staticmethod
    def _shard_index(name: str) -> int:
        mid = name[name.index(_SHARD_KEY_MARK) + len(_SHARD_KEY_MARK) :]
        return int(mid.split(".", 1)[0])

    def close(self) -> None:
        if self._closed:
            return
        self._shards = []
        self._row_starts = torch.empty(0, dtype=torch.int64)
        self._row_ends = torch.empty(0, dtype=torch.int64)
        keeps, self._keep = self._keep, []
        for item in keeps:
            for obj in item if isinstance(item, tuple) else (item,):
                close = getattr(obj, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
        self._closed = True

    def __enter__(self) -> NGramBank:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def split_global_ids(self, global_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global padded-table rows to ``(shard_idx, local_row)``, same shape."""
        if self._closed:
            raise RuntimeError("NGramBank is closed")
        ids = global_ids.to(dtype=torch.int64, device="cpu")
        # right=True: first end strictly greater than id → owning shard.
        shard_idx = torch.searchsorted(self._row_ends, ids, right=True)
        local = ids - self._row_starts[shard_idx]
        return shard_idx, local

    def gather_fp8(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Gather raw fp8 rows. ``global_ids`` any integer shape → ``[..., head_dim]``."""
        if self._closed:
            raise RuntimeError("NGramBank is closed")
        ids = global_ids.to(dtype=torch.int64, device="cpu")
        flat = ids.reshape(-1)
        if flat.numel() and (bool((flat < 0).any()) or bool((flat >= self.padded_vocab).any())):
            raise IndexError(
                f"global id outside [0, {self.padded_vocab}): "
                f"min={int(flat.min())} max={int(flat.max())}"
            )
        out = torch.empty((flat.numel(), self.head_dim), dtype=self._shards[0].table.dtype)
        if flat.numel() == 0:
            return out.view(*ids.shape, self.head_dim)
        shard_idx, local = self.split_global_ids(flat)
        # Per-shard masked scatter: mmap views are immutable and published in
        # open(); each iteration only reads one shard. Unique-shard loop beats
        # a 128-way Python gather when a decode tile hits 1–2 shards.
        for s in shard_idx.unique().tolist():
            s = int(s)
            if s < 0 or s >= self.num_shards:
                raise IndexError(f"shard index {s} out of range [0, {self.num_shards})")
            mask = shard_idx == s
            loc = local[mask]
            if torch.any(loc < 0) or torch.any(loc >= self._shards[s].n_rows):
                raise IndexError(f"local row out of range in shard {s}")
            out[mask] = self._shards[s].table[loc]
        return out.view(*ids.shape, self.head_dim)

    def _dequant(self, fp8_rows: torch.Tensor) -> torch.Tensor:
        """Per-tensor scale: ``bf16 = fp8.to(bf16) * weight_scale`` (broadcast)."""
        scale = self.weight_scale.to(device="cpu", dtype=torch.bfloat16).reshape(
            *([1] * fp8_rows.ndim)
        )
        return fp8_rows.to(torch.bfloat16) * scale

    def lookup(self, global_ids: torch.Tensor) -> torch.Tensor:
        """CPU gather + dequant. ``LongTensor[B, S, 16] -> bf16 [B, S, 16, 160]``."""
        return self._dequant(self.gather_fp8(global_ids))

    def lookup_to(
        self,
        global_ids: torch.Tensor,
        device: torch.device | str | int | None,
        *,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """``lookup`` then optional pin-stage + async copy to ``device``."""
        cpu = self.lookup(global_ids)
        if _device_is_cpu(device):
            return cpu
        # Per-call pin (tile is ~5 KB bf16): avoids a shared staging race if
        # two decode hooks overlap. copy_ completes on CPU before the
        # non_blocking H2D is issued, so dest is not observed until the
        # caller's stream sees this copy (default stream: immediately).
        try:
            pinned = torch.empty_like(cpu, pin_memory=True)
            pinned.copy_(cpu)
            src = pinned
        except RuntimeError:
            src = cpu
        return src.to(device, non_blocking=non_blocking)


__all__ = ["NGramBank"]
