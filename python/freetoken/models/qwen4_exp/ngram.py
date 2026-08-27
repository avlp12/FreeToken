from __future__ import annotations

import json
import mmap
import os
import threading
import time as _time
import warnings

import safetensors
import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP
from freetoken.utils import init_logger

from .hyperconnect import grouped_rms_norm

logger = init_logger(__name__)

# ---------- n-gram gather cost probe (temporary instrumentation) ----------
# FREETOKEN_NGRAM_TIMING=N     -> after FREETOKEN_NGRAM_TIMING_SKIP warmup decode
#   steps, wall-clock (time.perf_counter -- this path is host-side CPU, no CUDA
#   involved) the next N decode-step n-gram gathers (both the gather() call alone
#   and precompute_decode_ngram()'s total, which also includes the hash compute
#   and the state bookkeeping) and log mean/median/min/max. Every prefill-path
#   gather is logged individually (one call per request per chunk, so volume is
#   bounded by the benchmark, not by decode-step count).
# FREETOKEN_NGRAM_STUB=1       -> replace the real mmap gather with a constant
#   zeros() return of the correct shape. Output is WRONG when this is set -- this
#   is a COST PROBE ONLY, never a correctness path. The wall-time delta between a
#   stub run and a real run at the same step count is the gather's true cost
#   including page-fault effects.
_NGRAM_TIMING = int(os.environ.get("FREETOKEN_NGRAM_TIMING", "0") or 0)
_NGRAM_TIMING_SKIP = int(os.environ.get("FREETOKEN_NGRAM_TIMING_SKIP", "8") or 0)
_NGRAM_STUB = int(os.environ.get("FREETOKEN_NGRAM_STUB", "0") or 0)

# Reference: transformers modeling_qwen4_exp (Qwen4ExpTextNGramEmbedding /
# Qwen4ExpTextPLELayer), read 2026-08-26. Faithful naive port:
#   * hash n-gram ids = XOR of (shifted token id * per-position multiplier), one prime
#     vocab per head, offsets into ONE big embedding table;
#   * the multipliers / head vocab sizes / head offsets are checkpoint BUFFERS -- we load
#     them from the checkpoint instead of re-deriving from the seed;
#   * PLE: key per stream + shared value from the n-gram embedding, signed-sqrt dot gate
#     against the normed streams, plus a dilated depthwise causal conv (dilation =
#     ngram_size) over the normed gated value. Output adds to the 4-stream hidden.
#
# The table (~51 GB, fp8 + one global scale, 128 shards) is deliberately NOT streamed
# through iter_weights: shards are memory-mapped from the checkpoint dir (host-resident,
# page-cache managed) and rows are gathered per token -- 16 rows x 320 B per token, so
# decode traffic is trivial. Performance work (pinning, GPU-side cache) is a later phase.

_PLE_PREFIX = "ple.ple_embedding."

# GGUF IQ4_NL support (Task: source the n-gram/PLE table from the UD-Q4_K_XL GGUF's
# `per_layer_token_embd.weight` instead of the fp8 safetensors release, to shrink the
# FTW's resident working set: 47.68 GiB (fp8) -> 26.82 GiB (IQ4_NL). Verified against
# the real GGUF via freetoken.models.gguf.reader.iter_gguf_tensors: torch_shape=
# (320001536, 160), ggml_type=20 (IQ4_NL), row_bytes=90 -- i.e. the SAME 320,001,536-row
# table as the fp8 source (128 shards x 2,500,012 rows/shard = 320,001,536, confirmed
# against the real fp8 checkpoint's shard_0.weight shape), just packed 5 blocks/row
# (head_dim=160, QK4_NL=32) instead of 1 fp8 byte/element.
#
# Block spec (llama.cpp / ggml GGUF format, this repo's own copy in
# freetoken/kernel/csrc/gguf/ggml-common.h + dequantize.cuh):
#   struct block_iq4_nl { half d; uint8_t qs[16]; };  // 18 bytes / 32 elements
#   y[k]    = d * kvalues_iq4nl[qs[k] & 0xf]   for k in [0, 16)
#   y[k+16] = d * kvalues_iq4nl[qs[k] >> 4]    for k in [0, 16)
# A row of head_dim=160 is 5 independent blocks (5*18=90 bytes), matching the GGUF
# reader's measured row_bytes=90 exactly.
_KVALUES_IQ4NL = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.float32,
)
_IQ4NL_BLOCK_ELEMS = 32
_IQ4NL_BLOCK_BYTES = 18


def _dequantize_iq4nl_rows(
    raw: torch.Tensor, head_dim: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """``[N, row_bytes]`` packed GGML IQ4_NL rows (uint8) -> ``[N, head_dim]`` dequantized.

    Pure function, no state -- see the module-level comment above for the block spec
    this mirrors (this repo's own dequantize.cuh, same LUT and bit layout).
    """
    assert head_dim % _IQ4NL_BLOCK_ELEMS == 0, head_dim
    n_blocks = head_dim // _IQ4NL_BLOCK_ELEMS
    n = raw.shape[0]
    assert raw.shape[1] == n_blocks * _IQ4NL_BLOCK_BYTES, (raw.shape, head_dim)
    blocks = raw.reshape(n, n_blocks, _IQ4NL_BLOCK_BYTES)
    d = blocks[:, :, 0:2].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    qs = blocks[:, :, 2:2 + _IQ4NL_BLOCK_ELEMS // 2]  # [n, n_blocks, 16] uint8
    lut = _KVALUES_IQ4NL.to(raw.device)
    lo = lut[(qs & 0x0F).long()]  # [n, n_blocks, 16]
    hi = lut[(qs >> 4).long()]  # [n, n_blocks, 16]
    out = torch.empty(n, n_blocks, _IQ4NL_BLOCK_ELEMS, dtype=torch.float32, device=raw.device)
    out[:, :, : _IQ4NL_BLOCK_ELEMS // 2] = lo * d.unsqueeze(-1)
    out[:, :, _IQ4NL_BLOCK_ELEMS // 2 :] = hi * d.unsqueeze(-1)
    return out.reshape(n, head_dim).to(out_dtype)


class _NGramTable:
    """Lazy, mmap-backed reader of the sharded n-gram embedding table.

    Dual path: if ``model.safetensors.index.json`` is present, keep the original
    safetensors ``safe_open`` / ``get_tensor`` mmap (always fp8 -- the fp8 release
    is the only safetensors source). Otherwise parse the FTW index
    (``freetoken_weight.json``) and mmap ``kind="ngram"`` (fp8 passthrough) or
    ``kind="ngram_iq4nl"`` (GGUF-sourced, packed IQ4_NL) tensors in place.
    ``iter_ftw_ngrams`` materializes every n-gram tensor (~27-48 GiB depending on
    source) so it is not used: this path only records ``(file, offset, shape, dtype)``
    and gathers the requested rows.
    """

    def __init__(self, model_path: str, layer_idx: int, split_parts: int):
        self.model_path = model_path
        self.key_base = (
            f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
        )
        self.split_parts = split_parts
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        self._handles: dict[str, object] = {}
        self._shards: list[torch.Tensor | None] = [None] * split_parts
        # Guards the lazy first-touch population of _handles / _shards above. No
        # caller on this branch actually contends on it today (gather() is only ever
        # invoked from the single engine thread here) -- it is included defensively
        # so any future concurrent consumer of this table (e.g. a background
        # prefetch thread) gets an already-correct primitive instead of re-deriving
        # it. RLock, not Lock: _shard() holds it while calling _tensor(), which (on
        # the non-FTW / safetensors path) acquires it again on the SAME thread -- a
        # plain Lock self-deadlocks there the first time anything calls in through
        # this lock reentrantly (caught during development of a since-shelved
        # concurrent-prefetch experiment, by test_qwen4_ple_ngram_hoist.py, which
        # runs against the real FP8/safetensors checkpoint and hung indefinitely
        # until this was RLock -- landed here on its own merits since the failure
        # mode is easy to reintroduce and easy to miss without a test that happens
        # to exercise reentry).
        self._lock = threading.RLock()
        self._ftw_locs: dict[str, tuple] | None = None
        self._iq4nl = False
        if os.path.isfile(index_path):
            with open(index_path) as fh:
                self._weight_map = json.load(fh)["weight_map"]
        else:
            self._weight_map = {}
            self._ftw_locs = self._load_ftw_ngram_locs()
        if self._iq4nl:
            # Scale is per-block (block_iq4_nl.d), already applied by
            # _dequantize_iq4nl_rows -- no separate global scale tensor exists for
            # this source, unlike the fp8 release's single weight_scale.
            self.weight_scale = 1.0
        else:
            self.weight_scale = self._tensor(f"{self.key_base}.weight_scale").float().item()
        if self._ftw_locs is not None:
            _file, _off, shape, _dt, _nb = self._ftw_locs[f"{self.key_base}.shard_0.weight"]
            self.rows_per_shard = int(shape[0])
            if self._iq4nl:
                # shape is the PACKED byte layout [rows, row_bytes] (dtype=uint8);
                # head_dim is derived algebraically from row_bytes, not read directly
                # (row_bytes=90 -> 5 IQ4_NL blocks -> head_dim=160).
                row_bytes = int(shape[1])
                assert row_bytes % _IQ4NL_BLOCK_BYTES == 0, row_bytes
                self.head_dim = (row_bytes // _IQ4NL_BLOCK_BYTES) * _IQ4NL_BLOCK_ELEMS
            else:
                self.head_dim = int(shape[1])
        else:
            shard0 = self._shard(0)
            self.rows_per_shard = shard0.shape[0]
            self.head_dim = shard0.shape[1]

    def _load_ftw_ngram_locs(self) -> dict[str, tuple]:
        """Index FTW ``kind=ngram``/``kind=ngram_iq4nl`` tensors without reading
        their payloads. Sets ``self._iq4nl`` as a side effect, from whichever kind
        the checkpoint's own shard_0 entry actually has (a single FTW carries only
        one ngram source -- convert.py never mixes the two)."""
        from freetoken.checkpoint.ftw import INDEX_NAME

        index_path = os.path.join(self.model_path, INDEX_NAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"neither model.safetensors.index.json nor {INDEX_NAME} "
                f"under {self.model_path}"
            )
        with open(index_path) as fh:
            index = json.load(fh)
        shards = sorted(index["shards"], key=lambda s: s["global_off"])
        locs: dict[str, tuple] = {}
        kinds: dict[str, str] = {}
        for t in index["tensors"]:
            kind = t.get("kind")
            if kind not in ("ngram", "ngram_iq4nl"):
                continue
            kinds[t["name"]] = kind
            pieces: list[tuple[str, int, int]] = []
            remaining = int(t["nbytes"])
            pos = int(t["global_off"])
            for sh in shards:
                s0 = int(sh["global_off"])
                s1 = s0 + int(sh["nbytes"])
                if pos >= s1 or remaining <= 0:
                    continue
                if pos < s0:
                    raise ValueError(f"FTW gap locating {t['name']}")
                take = min(remaining, s1 - pos)
                pieces.append((sh["file"], pos - s0, take))
                pos += take
                remaining -= take
            if remaining:
                raise ValueError(f"FTW range exceeds shards for {t['name']}")
            if len(pieces) != 1:
                raise ValueError(
                    f"ngram tensor {t['name']} spans FTW files: {pieces}"
                )
            file, file_off, nbytes = pieces[0]
            dtype = getattr(torch, t["dtype"])
            locs[t["name"]] = (file, file_off, tuple(t["shape"]), dtype, nbytes)
        for i in range(self.split_parts):
            key = f"{self.key_base}.shard_{i}.weight"
            if key not in locs:
                raise KeyError(f"FTW ngram missing {key}")
        shard0_kind = kinds.get(f"{self.key_base}.shard_0.weight")
        self._iq4nl = shard0_kind == "ngram_iq4nl"
        return locs

    def _ftw_u8(self, filename: str) -> torch.Tensor:
        """Read-only mmap of one FTW shard file as a uint8 view (no copy)."""
        cached = self._handles.get(filename)
        if cached is not None:
            return cached[-1]
        with self._lock:  # double-checked: another thread may have won the race
            cached = self._handles.get(filename)
            if cached is not None:
                return cached[-1]
            path = os.path.join(self.model_path, filename)
            fh = open(path, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="The given buffer is not writable"
                    )
                    u8 = torch.frombuffer(mm, dtype=torch.uint8)
                self._handles[filename] = (fh, mm, u8)
            except (ValueError, RuntimeError, TypeError):
                import numpy as np

                arr = np.memmap(path, dtype=np.uint8, mode="r")
                u8 = torch.from_numpy(arr)
                self._handles[filename] = (fh, mm, arr, u8)
            return u8

    def _tensor(self, key: str) -> torch.Tensor:
        if self._ftw_locs is not None:
            file, file_off, shape, dtype, nbytes = self._ftw_locs[key]
            raw = self._ftw_u8(file)[file_off:file_off + nbytes]
            viewed = raw.view(dtype)
            return viewed if not shape else viewed.view(*shape)
        shard_file = self._weight_map[key]
        h = self._handles.get(shard_file)
        if h is None:
            with self._lock:
                h = self._handles.get(shard_file)
                if h is None:
                    h = safetensors.safe_open(
                        os.path.join(self.model_path, shard_file), framework="pt", device="cpu"
                    )
                    self._handles[shard_file] = h
        return h.get_tensor(key)

    def _shard(self, i: int) -> torch.Tensor:
        t = self._shards[i]
        if t is None:
            with self._lock:
                t = self._shards[i]
                if t is None:
                    t = self._tensor(f"{self.key_base}.shard_{i}.weight")
                    self._shards[i] = t
        return t

    def buffer(self, name: str) -> torch.Tensor:
        """Small int64 sidecar buffers: layer_multipliers / ngram_heads_offsets /
        ngram_heads_vocab_sizes (loaded verbatim, never dtype-cast)."""
        key = self.key_base.rsplit(".", 1)[0] + "." + name  # ...ple.ple_embedding.<name>
        return self._tensor(key).clone()

    def _rows(self, shard_i: int, local: torch.Tensor) -> torch.Tensor:
        """Rows ``[N, head_dim]``, real values (fp8 already-scaled-by-gather()'s
        caller, or IQ4_NL already fully dequantized here). Source path uses the
        cached mmap shard; FTW indexes the uint8 file view so only the selected row
        bytes are copied."""
        if self._ftw_locs is None:
            return self._shard(shard_i).index_select(0, local)
        file, file_off, shape, dtype, nbytes = self._ftw_locs[
            f"{self.key_base}.shard_{shard_i}.weight"
        ]
        # row_nbytes is correct for BOTH sources: fp8 shape[1]=head_dim,
        # dtype.itemsize=1 -> head_dim bytes/row; IQ4_NL shape[1]=row_bytes (packed),
        # dtype=uint8, itemsize=1 -> row_bytes bytes/row (shape[1] IS already the
        # packed byte width in that case, not an element count).
        row_nbytes = int(shape[1]) * dtype.itemsize
        table = self._ftw_u8(file)[file_off:file_off + nbytes].view(
            int(shape[0]), row_nbytes
        )
        gathered = table.index_select(0, local)
        if self._iq4nl:
            return _dequantize_iq4nl_rows(gathered, self.head_dim, torch.float32)
        return gathered.view(dtype).view(local.numel(), int(shape[1]))

    def gather(self, ids: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """ids [*] int64 global rows -> [*, head_dim] dequantized embeddings (CPU gather,
        returned on ids' original device)."""
        device = ids.device
        if _NGRAM_STUB:
            # COST PROBE ONLY -- see module docstring. Skips the real mmap gather
            # (host sync, data-dependent shard loop, unpinned H2D) entirely and
            # returns zeros of the correct shape. Output is WRONG under this flag.
            return torch.zeros(*ids.shape, self.head_dim, dtype=out_dtype, device=device)
        flat = ids.reshape(-1).cpu()
        shard_idx = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        local = flat - shard_idx * self.rows_per_shard
        out = torch.empty(flat.shape[0], self.head_dim, dtype=out_dtype)
        for s in torch.unique(shard_idx).tolist():
            mask = shard_idx == s
            rows = self._rows(int(s), local[mask])
            out[mask] = rows.to(out_dtype) * self.weight_scale
        return out.to(device).reshape(*ids.shape, self.head_dim)


def _shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos: int) -> torch.Tensor:
    """Reference-faithful: shift right by ``shift`` WITHOUT crossing eos boundaries
    (positions whose segment started after the source fall back to eos)."""
    if shift == 0:
        return token_ids
    batch, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos, positions, torch.full_like(positions, -1))
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos))


class Qwen4PLELayer(BaseOP):
    """Per-Layer (n-gram) Embedding block, attached to one decoder layer (layer_idx 1
    for the released checkpoint). Adds hashed lexical features to the 4-stream hidden.

    Weight keys (under ``model.layers.<L>.ple.``): key_proj / value_proj (bf16 linears),
    norm_key / norm_query / norm_conv (grouped Gemma-style norms, +1 baked by the
    loader), conv1d (depthwise, dilation=ngram_size). The ngram_embedding subtree is
    read directly from the checkpoint by ``_NGramTable``.

    Recurrent state per request slot (mirrors the GDN conv-state pattern, own buffers):
      * token history: last (ngram_size-1) input ids
      * conv tail: last (kernel-1)*dilation normed-gated-value steps
    """

    def __init__(self, config, layer_idx: int):
        args = config.qwen4_args
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        self.eps = config.rms_norm_eps
        self.eos = args.eos_token_id
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * args.heads_per_ngram
        self.ple_embed_dim = args.ple_embed_dim
        self.conv_kernel = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.state_len = (self.conv_kernel - 1) * self.conv_dilation
        hc_hidden = self.hc_count * self.hidden_size

        self.key_proj = _W(hc_hidden, args.ple_embed_dim)
        self.value_proj = _W(self.hidden_size, args.ple_embed_dim)
        self.norm_key = _N(hc_hidden)
        self.norm_query = _N(hc_hidden)
        self.norm_conv = _N(hc_hidden)
        self.conv1d = _Conv(hc_hidden, self.conv_kernel)

        assert args.model_path, "qwen4_args.model_path required for the n-gram table"
        self._table = _NGramTable(args.model_path, layer_idx, args.split_ngram_parts)
        # Checkpoint buffers (int64) -- authoritative over re-derivation from the seed.
        # Underscore names: BaseOP registers every non-underscore Tensor attribute as a
        # REQUIRED state-dict entry, but these come from the n-gram table source (not
        # iter_weights), so they must stay out of load_state_dict.
        self._layer_multipliers = self._table.buffer("layer_multipliers")
        self._head_vocab_sizes = self._table.buffer("ngram_heads_vocab_sizes")
        self._head_offsets = self._table.buffer("ngram_heads_offsets")
        # Per-slot recurrent state, allocated lazily (slot count comes from the pool).
        self._tok_hist: torch.Tensor | None = None
        self._conv_tail: torch.Tensor | None = None
        # Per-slot n-gram embedding, filled by ``precompute_decode_ngram`` (eager,
        # pre-graph) and read by the (possibly captured) decode forward. Stable
        # address across replays -- see precompute_decode_ngram's docstring.
        self._ngram_embed_buf: torch.Tensor | None = None
        # Cost-probe instrumentation state (env-gated, see module docstring).
        self._ngram_timing_left = _NGRAM_TIMING
        self._ngram_timing_skip = _NGRAM_TIMING_SKIP
        self._decode_gather_ms: list[float] = []
        self._decode_total_ms: list[float] = []

    # ---------- state pool ----------

    def _ensure_state(self, slots: int, device, dtype):
        if self._tok_hist is None:
            self._tok_hist = torch.full(
                (slots, self.context_len), self.eos, dtype=torch.long, device=device
            )
            self._conv_tail = torch.zeros(
                slots, self.hc_count * self.hidden_size, self.state_len,
                dtype=dtype, device=device,
            )
            self._ngram_embed_buf = torch.zeros(
                slots, self.ple_embed_dim, dtype=dtype, device=device,
            )
            if self._layer_multipliers.device != device:
                self._layer_multipliers = self._layer_multipliers.to(device)
                self._head_vocab_sizes = self._head_vocab_sizes.to(device)
                self._head_offsets = self._head_offsets.to(device)

    def _reset_slots(self, idx: torch.Tensor):
        self._tok_hist.index_fill_(0, idx, self.eos)
        self._conv_tail.index_fill_(0, idx, 0.0)
        self._ngram_embed_buf.index_fill_(0, idx, 0.0)

    # ---------- n-gram hashing ----------

    def _ngram_embed(
        self, history: torch.Tensor, out_len: int, dtype, *, timing_tag: str | None = None
    ) -> torch.Tensor:
        """history [B, context_len + T] -> [B, out_len(=T), ple_embed_dim].

        ``timing_tag`` ("decode" / "prefill") is a cost-probe hook only (see module
        docstring): when set and ``FREETOKEN_NGRAM_TIMING`` is on, wall-clocks the
        ``_table.gather`` call alone (excludes the hash compute above it) and
        records/logs it. No effect on the returned value in any case.
        """
        shifted = [
            _shift_right_ignore_eos(history, s, self.eos) for s in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self._layer_multipliers[0]
            for pos in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * self._layer_multipliers[pos])
            sizes = self._head_vocab_sizes[start:end]
            offs = self._head_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1))
            blocks.append(ids + offs.view(1, 1, -1))
        ids = torch.cat(blocks, dim=-1)[:, -out_len:]  # [B, T, ngram_heads]
        do_time = (
            _NGRAM_TIMING > 0
            and timing_tag is not None
            and (timing_tag != "decode" or (self._ngram_timing_skip == 0 and self._ngram_timing_left > 0))
        )
        if do_time:
            _g0 = _time.perf_counter()
            emb = self._table.gather(ids, dtype)  # [B, T, heads, head_dim]
            _gms = (_time.perf_counter() - _g0) * 1000.0
            n_rows = ids.shape[0] * ids.shape[1]  # batch * out_len (tokens gathered)
            if timing_tag == "decode":
                self._decode_gather_ms.append(_gms)
                logger.info_rank0(
                    "[ngram-timing] decode step=%d layer=%d gather_ms=%.4f",
                    len(self._decode_gather_ms), self.layer_idx, _gms,
                )
            elif timing_tag == "prefill":
                logger.info_rank0(
                    "[ngram-timing] prefill layer=%d tokens=%d rows=%d gather_ms=%.4f "
                    "us_per_row=%.4f",
                    self.layer_idx, out_len, n_rows, _gms,
                    (_gms * 1000.0) / max(n_rows, 1),
                )
        else:
            emb = self._table.gather(ids, dtype)  # [B, T, heads, head_dim]
        return emb.flatten(-2)

    # ---------- CUDA-graph hoist: eager pre-replay n-gram gather ----------

    def precompute_decode_ngram(self, batch) -> None:
        """Eagerly compute this decode step's n-gram embedding and advance the
        per-slot token-history state, BEFORE any CUDA-graph replay touches this
        layer. Called once per decode step by the engine, for both the captured and
        the eager decode path (see ``engine.forward_batch`` / ``Qwen4ExpForCausalLM.
        precompute_ngram_embed``).

        Why this cannot live inside the captured decode forward: ``_NGramTable.
        gather()`` syncs device->host (``.cpu()``), loops over a data-dependent set
        of shard ids (``torch.unique(...).tolist()``), and copies back with an
        unpinned H2D -- all three are illegal during CUDA graph capture (the first
        raises "Cannot copy between CPU and CUDA tensors during CUDA graph capture
        unless the CPU tensor is pinned").

        Pinning the staging buffer does NOT fix this. A CUDA graph replays only the
        GPU ops it recorded; host-side Python does not re-execute on replay. A
        captured host-side gather would run once at capture time and every replay
        after would silently reuse that one stale embedding -- wrong output, no
        error, the worst failure mode here.

        The PLE embedding is a pure function of recent token ids (see
        ``_shift_right_ignore_eos``: only ``token_ids`` and ``eos`` feed the hash;
        the multipliers / offsets / vocab sizes are constant checkpoint buffers), so
        it can be computed here, outside the graph, and hoisted into a stable-address
        per-slot buffer -- exactly the pattern this module already uses for
        ``_tok_hist`` / ``_conv_tail`` recurrent state (mirrors the GDN conv-state
        pattern this codebase captures today). The captured decode branch then only
        does a device-side ``index_select`` against ``_ngram_embed_buf``, which is
        capture-safe: a dynamic index driven by a device tensor needs no host sync,
        only the *value* of ``idx`` may vary between replays, not the traced op.

        Decode needs history: with ``ngram_size=3`` the hash at step *t* needs
        tokens *t-2, t-1, t*, but only the newest token is fed to the model at
        decode. ``_tok_hist[idx]`` carries the previous ``context_len`` (=2) ids per
        slot from the last step; this reads it (old state) before overwriting it
        with the new tail, same order as the pre-hoist decode forward used.
        """
        _timing = (
            _NGRAM_TIMING > 0 and self._ngram_timing_skip == 0 and self._ngram_timing_left > 0
        )
        _t0 = _time.perf_counter() if _timing else 0.0
        ctx = get_global_ctx()
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, batch.input_ids.device)
            batch.fla_metadata = fla
        li = pool.local_index(self.layer_idx)
        slots = pool.conv_states[li].shape[0]
        self._ensure_state(slots, batch.input_ids.device, self.key_proj.weight.dtype)

        idx = fla.cache_indices
        idx64 = idx.long()
        input_ids = batch.input_ids.to(torch.long)
        hist = torch.cat([self._tok_hist[idx], input_ids.view(-1, 1)], dim=-1)  # [B, context_len+1]
        emb = self._ngram_embed(
            hist, 1, self._ngram_embed_buf.dtype, timing_tag="decode"
        )[:, 0]  # [B, ple_embed_dim]
        self._ngram_embed_buf.index_copy_(0, idx64, emb)
        self._tok_hist.index_copy_(0, idx64, hist[:, 1:])

        # Cost-probe bookkeeping (see module docstring) -- runs AFTER the skip-warmup
        # steps, times the whole eager hoist (hash + gather + state bookkeeping) for
        # FREETOKEN_NGRAM_TIMING steps, then logs a summary once and stops.
        if _NGRAM_TIMING > 0 and self._ngram_timing_skip > 0:
            self._ngram_timing_skip -= 1
        elif _timing:
            self._decode_total_ms.append((_time.perf_counter() - _t0) * 1000.0)
            self._ngram_timing_left -= 1
            if self._ngram_timing_left == 0:
                self._flush_ngram_timing()

    def _flush_ngram_timing(self) -> None:
        def _stats(vals: list[float], label: str) -> None:
            if not vals:
                return
            v = sorted(vals)
            n = len(v)
            logger.info_rank0(
                "[ngram-timing] decode %s over %d steps (layer=%d): mean=%.4fms "
                "median=%.4fms min=%.4fms max=%.4fms",
                label, n, self.layer_idx, sum(v) / n, v[n // 2], v[0], v[-1],
            )

        _stats(self._decode_gather_ms, "gather-only")
        _stats(self._decode_total_ms, "precompute-total")

    # ---------- PLE core (matches the reference forward) ----------

    def _ple_core(self, x4: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """x4 [N, hc*H] current streams, emb [N, ple_embed_dim] -> gated value [N, hc*H]
        (pre-conv part) -- caller runs the causal conv on the normed copy."""
        key = F.linear(emb, self.key_proj.weight)
        key_normed = grouped_rms_norm(key, self.norm_key.weight, self.hidden_size, self.eps)
        key_normed = key_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        value = F.linear(emb, self.value_proj.weight)
        query_normed = grouped_rms_norm(
            x4, self.norm_query.weight, self.hidden_size, self.eps
        ).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / (self.hidden_size ** 0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)  # [N, hc, H]
        return gated.flatten(-2)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight  # [hc*H, 1, K]

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        """Returns the PLE delta to ADD to the streams: [N, hc*H]."""
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, x4.device)
            batch.fla_metadata = fla
        li = pool.local_index(self.layer_idx)
        slots = pool.conv_states[li].shape[0]
        self._ensure_state(slots, x4.device, x4.dtype)

        if batch.is_decode:
            # The n-gram embedding for this step was already computed and parked in
            # _ngram_embed_buf by precompute_decode_ngram, which the engine runs
            # eagerly before any CUDA-graph replay reaches this forward (host-mmap
            # table gather is illegal inside a captured graph -- see that method's
            # docstring). This index_select is a plain device-tensor gather:
            # capture-safe, no host sync, mirrors the existing _tok_hist/_conv_tail
            # per-slot state reads directly below.
            idx = fla.cache_indices  # [B] slot per row
            idx64 = idx.long()
            emb = self._ngram_embed_buf.index_select(0, idx64)  # [B, ple_embed_dim]
            gated = self._ple_core(x4, emb)  # [B, hc*H]
            normed = grouped_rms_norm(gated, self.norm_conv.weight, self.hidden_size, self.eps)
            # dilated causal conv, single step: taps at t-9, t-6, t-3, t (K=4, d=3)
            tail = self._conv_tail[idx]  # [B, C, 9]
            w = self._conv_weight().squeeze(1)  # [C, K]
            conv = w[:, -1] * normed
            for k in range(1, self.conv_kernel):
                conv = conv + w[:, -1 - k] * tail[:, :, -k * self.conv_dilation].to(normed.dtype)
            conv = F.silu(conv)
            # state update: append the current normed step
            new_tail = torch.cat([tail[:, :, 1:], normed.unsqueeze(-1).to(tail.dtype)], dim=-1)
            # index_copy_ requires a long index (fla.cache_indices is int32, same as
            # every other CUDA-graph-safe index tensor in the engine; plain indexing
            # above (self._conv_tail[idx]) accepts int32 fine, but index_copy_'s ATen
            # kernel is stricter). Cast locally rather than widening cache_indices
            # itself, which other consumers may rely on staying int32.
            # (_tok_hist's own index_copy_ now happens in precompute_decode_ngram,
            # eagerly, alongside the gather that needed the pre-update history.)
            self._conv_tail.index_copy_(0, idx64, new_tail)
            return gated + conv

        # ---- prefill (varlen; per-request loop, correctness over speed in P0) ----
        input_ids = batch.input_ids.to(torch.long)
        if fla.fresh_state_indices is not None:
            self._reset_slots(fla.fresh_state_indices)
        cu = fla.cu_seqlens.tolist()
        out = torch.empty_like(x4)
        w = self._conv_weight()
        for r in range(len(cu) - 1):
            s, e = cu[r], cu[r + 1]
            slot = int(fla.cache_indices[r])
            ids_r = input_ids[s:e].view(1, -1)  # [1, T]
            hist = torch.cat([self._tok_hist[slot].view(1, -1), ids_r], dim=-1)
            emb = self._ngram_embed(hist, e - s, x4.dtype, timing_tag="prefill")[0]  # [T, D]
            gated = self._ple_core(x4[s:e], emb)  # [T, C]
            normed = grouped_rms_norm(gated, self.norm_conv.weight, self.hidden_size, self.eps)
            tail = self._conv_tail[slot]  # [C, 9]
            seq = torch.cat([tail.to(normed.dtype), normed.t()], dim=-1)  # [C, 9+T]
            conv = F.conv1d(
                seq.unsqueeze(0), w.to(normed.dtype), dilation=self.conv_dilation,
                groups=w.shape[0],
            )[0]  # [C, T]
            conv = F.silu(conv)
            out[s:e] = gated + conv.t()
            # persist state
            self._conv_tail[slot] = seq[:, -self.state_len:].to(self._conv_tail.dtype)
            self._tok_hist[slot] = hist[0, -self.context_len:]
        return out


class _W(BaseOP):
    def __init__(self, out_features: int, in_features: int):
        self.weight = torch.empty(out_features, in_features)


class _N(BaseOP):
    def __init__(self, dim: int):
        self.weight = torch.empty(dim)


class _Conv(BaseOP):
    def __init__(self, channels: int, kernel: int):
        self.weight = torch.empty(channels, 1, kernel)


__all__ = ["Qwen4PLELayer"]
