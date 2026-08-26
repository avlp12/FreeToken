"""Prefill routed-expert GEMM that decodes each expert ONCE per chunk.

Why
---
``ggml_moe_a8_vec``/``..._batched`` keep the expert in its packed GGUF blocks and
run the codebook dequant *inside* the dot product. That is exactly right for
decode -- the weight is read once and touched once -- and exactly wrong for
prefill, where an 8192-token chunk routes 49152 rows over 256 experts and every
expert therefore gets DECODED ~192 times per layer. Weight-reuse batching
(``FREETOKEN_MOE_PREFILL_BATCH``) cut the HBM traffic by N but not the ALU work:
one block still re-runs the IQ2_XS grid lookups for each of its N rows. Measured
at the production shape, the chunk's two GEMVs were 11.0 s of a 13.1 s chunk with
the memory traffic already amortized -- i.e. what is left is redundant DEQUANT.

What
----
Decode a TILE of experts once into a bf16 scratch matrix, then run one grouped
bf16 GEMM over the routed rows those experts own. Each expert is decoded exactly
once per chunk instead of ~24 times per layer, and the multiply itself moves onto
the tensor cores. Measured on an RTX 5090 at the production geometry, one layer's
two GEMMs over an 8192-token chunk: 248 ms -> 40 ms, 6.3x
(``benchmarks/bench_prefill_dequant_gemm.py``). Transient peak is 1.3 GiB at
tile 16, against roughly 8 GiB free while serving.

This CHANGES NUMERICS relative to the vec path, deliberately. The vec kernel
quantizes the ACTIVATION to q8_1 (8 bits per 32-element block) to feed the
integer dot product; here the activation stays bf16 end to end and only the
weight carries quantization error. Measured relative L2 difference against the
vec path is ~5e-3 on all three q2_k_ud types, essentially all of it the vec
path's activation quantization -- so this path is if anything the more accurate
of the two. See ``tests/kernels/test_dequant_gemm.py``.

Pitch, without touching the kernels
-----------------------------------
``ggml_dequantize`` has no pitch parameter: it walks ``m * n`` elements assuming
a tightly packed bank. The q2_k_ud banks are not tightly packed -- every row is
padded out to one shared IQ3_XXS-width pitch. But each row IS tightly packed from
its own base (that is precisely what ``row_pitch_bytes`` addressing means), with
the tail zero. So slicing the native prefix off every row,
``bank[e0:e1, :, :native_row_bytes].contiguous()``, yields a genuine tight bank
that ``ggml_dequantize`` reads correctly. ``native_row_bytes`` comes from
:func:`freetoken.kernel.gguf.ggml_type_row_bytes`, which is derived from the
``sizeof(block_q_t)`` the dequant kernels themselves compile against.

Grouped GEMM
------------
``torch._grouped_mm(a, b, offs=offs)`` with ``a`` ``[M, K]``, ``b`` ``[G, K, N]``
and ``offs`` an int32 tensor of CUMULATIVE END row offsets into ``a``. We want
``out[r] = A[r] @ W[e].T``, so ``b[g]`` must be ``W[e].T``. The dequantized tile
is ``[G, nrows, ncols]`` contiguous, and ``_grouped_mm`` accepts the
``transpose(1, 2)`` VIEW of it (column-major, non-contiguous) directly -- verified
bit-exact against per-group ``a @ b[g]`` on sm120 -- so no transpose is
materialized and the tile's peak memory is not doubled.

Host syncs
----------
The per-layer path must not stall the prefill copy pipeline, so everything here
is device-side except ONE small transfer per layer: the row boundaries of the
expert tiles. Tiles are fixed EXPERT ranges, so those boundaries are
``n_tiles + 1`` entries of the cumulative-count vector -- one pinned D2H copy of
a few hundred bytes, issued as soon as the counts exist and not waited on until
the caller has already enqueued the activation gather. Both GEMMs of a layer
share one plan (a routed-row index means the same thing in each), so it is one
transfer per layer, not per GEMM.
"""

from __future__ import annotations

import os
import warnings

import torch

# 16 experts: a 0.54 GB gate_up tile, 0.27 GB down. 32 is marginally faster at a
# 2048-token chunk and marginally SLOWER at 8192, for double the tile -- not a
# trade worth taking on a card whose free VRAM while serving is ~8 GiB.
_DEFAULT_TILE = 16

# Below this many tokens in the chunk, fall back to the vec path: the dequant is
# a FIXED ~14 ms per layer (all 256 experts, whatever the routing), so a chunk
# too small to amortize it loses.
#
# The isolated benchmark (``benchmarks/bench_prefill_dequant_gemm.py``) puts the
# crossover near 900 tokens, but that bench holds both banks RESIDENT. In the live
# server the expert bank is streaming over PCIe during the GEMM, and the dequant
# writes ~12.9 GB of bf16 per layer, so the two contend for HBM and the crossover
# moves out. Measured end-to-end on the real server (wall, dequant vs vec):
#   2073 tok  3.98s vs 4.45s  -> vec wins
#   2552 tok  4.33s vs 4.32s  -> tie
#   4124 tok  6.67s vs 4.47s  -> dequant 1.49x
#   6180 tok  9.85s vs 4.90s  -> dequant 2.01x
#   8220 tok 14.34s vs 6.50s  -> dequant 2.21x
# So the default sits just above the measured tie, which is what keeps small
# prompts from regressing. Trust the server number over the bench number.
_DEFAULT_MIN_TOKENS = 2560


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = int(raw.strip())
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not an integer; using the default {default}",
            RuntimeWarning,
            stacklevel=2,
        )
        return default
    if v < minimum:
        warnings.warn(
            f"{name}={v} is below the minimum {minimum}; using the default {default}",
            RuntimeWarning,
            stacklevel=2,
        )
        return default
    return v


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# Read once at import, like ``fused_q2_k_ud._PREFILL_BATCH``: the per-layer path
# must not pay an environment lookup, and the value must not shift mid-run
# underneath a captured CUDA graph.
ENABLED = _env_flag("FREETOKEN_PREFILL_DEQUANT_GEMM", True)
TILE = _env_int("FREETOKEN_PREFILL_DEQUANT_TILE", _DEFAULT_TILE)
MIN_TOKENS = _env_int("FREETOKEN_PREFILL_DEQUANT_MIN_TOKENS", _DEFAULT_MIN_TOKENS, minimum=0)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

# (qtype, ncols) -> native packed row bytes. Metadata only; tiny and unbounded
# only by the number of distinct bank geometries in the model (two).
_ROW_BYTES: dict[tuple[int, int], int] = {}
# Geometries whose pad tail has already been proven zero, so the check is paid
# once per bank shape rather than once per layer.
_PAD_CHECKED: set[tuple[int, int, int, int]] = set()
# Cleared by disable_pad_tail_check when the DEVICE bank stops being the right place
# to look for that evidence.
_PAD_CHECK_ENABLED = True


def disable_pad_tail_check(reason: str) -> None:
    """Stop asserting that a device bank's row tail is zero.

    Called by OffloadMoeCache once per-layer copy widths are live. Slots are shared
    across layers, and a layer whose native row IS the bank pitch has no padding at
    all, so once copies stop rewriting the whole row its real weights survive in the
    tail a narrower layer would call padding -- correct data that this check would
    reject. The evidence it provided moves to the host banks, where those bytes really
    are padding; see OffloadMoeCache._verify_host_pad_tails.
    """
    global _PAD_CHECK_ENABLED
    if _PAD_CHECK_ENABLED:
        from freetoken.utils import init_logger

        init_logger(__name__).info_rank0(
            f"prefill dequant-GEMM: device pad-tail check off ({reason})"
        )
    _PAD_CHECK_ENABLED = False


def native_row_bytes(qtype: int, ncols: int) -> int:
    """Packed byte width of one ``ncols``-element row of ``qtype``; 0 if unsupported."""
    key = (int(qtype), int(ncols))
    got = _ROW_BYTES.get(key)
    if got is None:
        from freetoken.kernel.gguf import ggml_type_row_bytes

        try:
            got = int(ggml_type_row_bytes(int(qtype), int(ncols)))
        except ValueError:
            got = 0
        _ROW_BYTES[key] = got
    return got


def supported(qtype: int, ncols: int, pitch: int) -> bool:
    """Whether a bank of this geometry can be sliced tight and dequantized."""
    if not hasattr(torch, "_grouped_mm"):
        return False
    n = native_row_bytes(qtype, ncols)
    return 0 < n <= pitch


def _check_pad_tail(bank: torch.Tensor, qtype: int, ncols: int, native: int) -> None:
    """Once per bank geometry: the bytes past ``native`` really are padding.

    The slice's correctness rests on each row being tightly packed from its own
    base with a zero tail. If ``native`` were too SMALL we would silently truncate
    real payload and decode garbage -- and garbage that still looks finite,
    because the IQ grids are total over their lookup tables. So prove the tail is
    dead before trusting the geometry. Costs one reduction on one expert slot,
    cached by (qtype, nrows, ncols, pitch) -- the same for all 43 layers of a
    bank, so a handful of checks per process.

    Off entirely once the cache declares per-layer copy widths -- see
    :func:`disable_pad_tail_check` for why the device bank stops being evidence.
    """
    if not _PAD_CHECK_ENABLED:
        return
    nrows, pitch = bank.shape[1], bank.shape[2]
    key = (int(qtype), int(nrows), int(ncols), int(pitch))
    if key in _PAD_CHECKED or native == pitch:
        _PAD_CHECKED.add(key)
        return
    tail = bank[0, :, native:]
    if bool(tail.any().item()):
        raise AssertionError(
            f"q2_k_ud bank tail is not padding: ggml type {qtype} with ncols "
            f"{ncols} packs {native} B/row, but bytes [{native}, {pitch}) of "
            f"expert slot 0 are nonzero. Either the derived row width is wrong "
            f"or the bank is not pitch-padded the way this path assumes."
        )
    _PAD_CHECKED.add(key)


# --------------------------------------------------------------------------- #
# routing plan
# --------------------------------------------------------------------------- #


class RoutePlan:
    """Routed rows grouped by expert, plus the expert-tile row boundaries.

    Built once per layer and used by BOTH GEMMs: the gate_up call runs with
    (tokens = T, top_k = 6) and the down call with (tokens = T * 6, top_k = 1),
    but a routed-row index means the same thing in each.

    Everything is device-side and sync-free except :meth:`tiles`, which reads
    ``n_tiles + 1`` int64s the constructor already started copying to pinned host
    memory. Call it AFTER enqueuing the activation gather so the GPU has work in
    flight across the wait.
    """

    __slots__ = ("order", "cum_i32", "routed", "num_experts", "tile", "_host", "_evt", "_tiles")

    def __init__(self, topk_ids: torch.Tensor, num_experts: int, tile: int):
        dev = topk_ids.device
        flat = topk_ids.reshape(-1).long()
        self.routed = flat.numel()
        self.num_experts = int(num_experts)
        self.tile = int(tile)

        # Stable, so the grouping is deterministic run to run and a failure
        # reproduces. ``order[i]`` is the routed-row index sitting at sorted
        # position i.
        self.order = torch.argsort(flat, stable=True)

        # ``scatter_add_`` rather than ``bincount``: bincount has to learn the max
        # element to size its output, which syncs.
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=dev)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        cum = torch.cumsum(counts, 0)  # [E] cumulative END offsets in sorted space
        self.cum_i32 = cum.to(torch.int32)

        # Tiles are fixed expert ranges, so their row boundaries are just
        # ``cum`` sampled at the last expert of each tile. One pinned D2H copy.
        edges = list(range(self.tile, self.num_experts, self.tile)) + [self.num_experts]
        idx = torch.tensor([e - 1 for e in edges], device=dev, dtype=torch.long)
        bounds = torch.cat([torch.zeros(1, dtype=cum.dtype, device=dev), cum[idx]])
        self._host = torch.empty(bounds.numel(), dtype=cum.dtype, pin_memory=True)
        self._host.copy_(bounds, non_blocking=True)
        self._evt = torch.cuda.Event()
        self._evt.record()
        self._tiles: tuple[tuple[int, int, int, int], ...] | None = None

    def tiles(self) -> tuple[tuple[int, int, int, int], ...]:
        """``(expert_lo, expert_hi, row_lo, row_hi)`` per tile, empty tiles dropped.

        The single host wait of the layer. Empty tiles are dropped here rather
        than skipped in the loop so a skewed routing that leaves most experts
        unrouted also skips their dequant.
        """
        if self._tiles is None:
            self._evt.synchronize()
            b = self._host.tolist()
            out = []
            for i, e0 in enumerate(range(0, self.num_experts, self.tile)):
                e1 = min(e0 + self.tile, self.num_experts)
                r0, r1 = int(b[i]), int(b[i + 1])
                if r1 > r0:
                    out.append((e0, e1, r0, r1))
            self._tiles = tuple(out)
        return self._tiles


# --------------------------------------------------------------------------- #
# the GEMM
# --------------------------------------------------------------------------- #


def grouped_expert_gemm(
    a_sorted: torch.Tensor,  # [R, ncols] activations already in expert-sorted order
    bank: torch.Tensor,  # [num_slots, nrows, pitch] uint8
    qtype: int,
    plan: RoutePlan,
) -> torch.Tensor:
    """``[R, nrows]``: for each sorted routed row, ``a_sorted[i] @ W[expert].T``.

    ``a_sorted`` must already be permuted into ``plan.order`` -- this function
    neither gathers nor scatters, because chaining two of these lets the whole
    layer stay in sorted order and pay the permutation exactly once.
    """
    from freetoken.kernel.gguf import ggml_dequantize

    nrows = bank.shape[1]
    ncols = a_sorted.shape[1]
    native = native_row_bytes(qtype, ncols)
    pitch = bank.shape[2]
    assert 0 < native <= pitch, (
        f"ggml type {qtype} packs {native} B for {ncols} columns, which does not "
        f"fit the bank pitch {pitch} B"
    )
    _check_pad_tail(bank, qtype, ncols, native)

    out = torch.empty((plan.routed, nrows), dtype=a_sorted.dtype, device=a_sorted.device)
    for e0, e1, r0, r1 in plan.tiles():
        n_t = e1 - e0
        # Tight bank: native prefix of every row, contiguous. Padding discarded.
        tight = bank[e0:e1, :, :native].contiguous()
        w = ggml_dequantize(
            tight.view(n_t * nrows, native), int(qtype), n_t * nrows, ncols, a_sorted.dtype
        )
        del tight  # free before the (much larger) tile is alive alongside it
        # [n_t, ncols, nrows] column-major VIEW -- _grouped_mm takes it as-is.
        b = w.view(n_t, nrows, ncols).transpose(1, 2)
        # Cumulative END offsets WITHIN the tile. r0 is a host int, so this stays
        # a single device subtraction -- no sync.
        offs = plan.cum_i32[e0:e1] - r0
        out[r0:r1] = torch._grouped_mm(a_sorted[r0:r1], b, offs=offs)
        del b, w
    return out


__all__ = [
    "ENABLED",
    "MIN_TOKENS",
    "TILE",
    "RoutePlan",
    "grouped_expert_gemm",
    "native_row_bytes",
    "supported",
]
