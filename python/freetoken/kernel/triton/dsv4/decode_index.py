"""Fused DSV4 decode-side index arithmetic.

Every decode step recomputes the same small family of derived addresses, once per
layer, out of three layer-invariant inputs -- the per-row position ``pos``, the
whole-history full-loc SNAPSHOT, and the 128-ring window slots. Composed in torch
that is a few dozen one-element int64 kernels per layer (``div_floor`` / ``add`` /
``mul`` / ``remainder`` / ``clamp`` / ``where`` / compare), and at 43 layers the
graph-node dispatch alone dominates the decode's fixed cost.

The three kernels here collapse that per-layer cluster:

* :func:`window_ring_ctx` -- the layer-INVARIANT 128-ring context (current window
  slot, previous window slot, and the window-first ring top-k), one launch per step;
* :func:`decode_index_ctx` -- everything a compressed layer derives from ``pos``
  alone, one launch per RATIO CLASS per step (DSV4-Flash has two: 4 and 128), since
  the derived values depend on the layer only through its compress ratio;
* :func:`cmp_topk_to_global` -- a layer's compressed picks resolved to global pool
  rows AND concatenated onto the window half, one launch per ratio-4 layer (the
  ratio-128 layers share a single launch, their picks being positional).

Semantics are bit-identical to the torch composition they replace: these are
ADDRESSES, so an off-by-one corrupts attention silently. In particular the integer
division is FLOOR division (``full_loc == -1`` must map to row ``-1``, not ``0``)
and the ring modulo is Python-style (``(pos - j) % win`` with ``pos < j``), neither
of which C truncation gives. Everything is fixed-shape with no host-side branch on
device data, so a captured decode graph replays unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1)).bit_length() if n > 1 else 1


@triton.jit
def _floordiv(a, b):
    """Floor division for int64 ``a`` by a POSITIVE constant ``b``.

    Written so it is correct whether Triton's ``//`` floors or truncates: the
    remainder is only ever adjusted when it came back negative, which a flooring
    ``//`` never produces for ``b > 0``.
    """
    q = a // b
    r = a - q * b
    return tl.where(r < 0, q - 1, q)


@triton.jit
def _pymod(a, b):
    """Python-style modulo (sign of the divisor) for int64 ``a``, positive ``b``."""
    return a - _floordiv(a, b) * b


@triton.jit
def _translate(f2w_ptr, full_loc, n, mask):
    """``full_to_window[full_loc]`` with torch's negative-index wrap.

    ``full_loc == -1`` is the pool's gather-only sentinel; torch resolves it to the
    map's trailing row, which is permanently ``-1``.
    """
    i = tl.where(full_loc < 0, full_loc + n, full_loc)
    ok = mask & (i >= 0) & (i < n)
    return tl.load(f2w_ptr + i, mask=ok, other=-1)


# --------------------------------------------------------------------------- #
# 1. layer-invariant 128-ring window context
# --------------------------------------------------------------------------- #
@triton.jit
def _window_ring_ctx_kernel(
    pos_ptr, rows_ptr, snap_ptr, f2w_ptr,
    ws_ptr, pws_ptr, topk_ptr,
    B, snap_stride, snap_w, f2w_n,
    WIN: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_W: tl.constexpr,
):
    b = tl.arange(0, BLOCK_B)
    mb = b < B
    pos = tl.load(pos_ptr + b, mask=mb, other=0)
    row = tl.load(rows_ptr + b, mask=mb, other=0)
    base = row * snap_stride

    # current position's window slot
    cur = tl.load(snap_ptr + base + pos, mask=mb & (pos < snap_w), other=-1)
    tl.store(ws_ptr + b, _translate(f2w_ptr, cur, f2w_n, mb), mask=mb)

    # previous position's window slot (the compressor's rolling-carry source)
    pm1 = tl.where(pos - 1 < 0, 0, pos - 1)
    prv = tl.load(snap_ptr + base + pm1, mask=mb & (pm1 < snap_w), other=-1)
    tl.store(pws_ptr + b, _translate(f2w_ptr, prv, f2w_n, mb), mask=mb)

    # ring slot j holds the latest p <= pos with p % win == j
    j = tl.arange(0, BLOCK_W)
    m2 = mb[:, None] & (j < WIN)[None, :]
    p = pos[:, None] - _pymod(pos[:, None] - j[None, :], WIN)
    pc = tl.where(p < 0, 0, p)
    fl = tl.load(snap_ptr + base[:, None] + pc, mask=m2 & (pc < snap_w), other=-1)
    ring = tl.where(p >= 0, _translate(f2w_ptr, fl, f2w_n, m2), -1)
    out = tl.where(j[None, :] <= pos[:, None], ring, -1)
    tl.store(topk_ptr + b[:, None] * WIN + j[None, :], out, mask=m2)


def window_ring_ctx(pos, rows, snap, full_to_window, win):
    """``(window_slots [B], prev_window_slots [B], window_slots_topk [B, 1, win])``."""
    B = pos.shape[0]
    ws = torch.empty(B, dtype=torch.int64, device=pos.device)
    pws = torch.empty(B, dtype=torch.int64, device=pos.device)
    topk = torch.empty(B, 1, win, dtype=torch.int64, device=pos.device)
    _window_ring_ctx_kernel[(1,)](
        pos, rows, snap, full_to_window,
        ws, pws, topk,
        B, snap.stride(0), snap.shape[1], full_to_window.numel(),
        WIN=win, BLOCK_B=_pow2(B), BLOCK_W=_pow2(win),
    )
    return ws, pws, topk


# --------------------------------------------------------------------------- #
# 2. per-ratio-class decode index context
# --------------------------------------------------------------------------- #
@triton.jit
def _decode_index_ctx_kernel(
    pos_ptr, rows_ptr, ws_ptr, pws_ptr, snap_ptr,
    idx_mod_ptr, should_ptr, ovl_slot_ptr,
    carry_prev_ptr, carry_cur_ptr,
    freq_idx_ptr, valid_ptr, cmp_counts_ptr,
    cmp_dst_attn_ptr, cmp_dst_idx_ptr,
    B, snap_stride, snap_w,
    RATIO: tl.constexpr, RING: tl.constexpr, P: tl.constexpr, CAP: tl.constexpr,
    CMP_BASE: tl.constexpr, IDX_BASE: tl.constexpr,
    HAS_IDX: tl.constexpr, OVERLAP: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_R: tl.constexpr,
):
    b = tl.arange(0, BLOCK_B)
    mb = b < B
    pos = tl.load(pos_ptr + b, mask=mb, other=0)
    row = tl.load(rows_ptr + b, mask=mb, other=0)
    ws = tl.load(ws_ptr + b, mask=mb, other=0)
    pws = tl.load(pws_ptr + b, mask=mb, other=0)

    # pos % ratio  /  (pos + 1) // ratio  /  (pos + 1) % ratio == 0
    idx_mod = pos - _floordiv(pos, RATIO) * RATIO
    tl.store(idx_mod_ptr + b, idx_mod, mask=mb)
    p1 = pos + 1
    valid = _floordiv(p1, RATIO)
    tl.store(valid_ptr + b, valid, mask=mb)
    should = (p1 - valid * RATIO) == 0
    tl.store(should_ptr + b, should.to(tl.int8), mask=mb)
    tl.store(cmp_counts_ptr + b, tl.minimum(valid, CAP).to(tl.int32), mask=mb)

    # the completed block's rope position: (pos + 1 - ratio).clamp_min(0)
    fi = p1 - RATIO
    tl.store(freq_idx_ptr + b, tl.where(fi < 0, 0, fi), mask=mb)
    if OVERLAP:
        tl.store(ovl_slot_ptr + b, RATIO + idx_mod, mask=mb)

    # per-row compress-state ring block: (window_slot // P) * ring_size + arange(R)
    r = tl.arange(0, BLOCK_R)
    m2 = mb[:, None] & (r < RING)[None, :]
    off = b[:, None] * RING + r[None, :]
    tl.store(carry_prev_ptr + off, _floordiv(pws, P)[:, None] * RING + r[None, :], mask=m2)
    tl.store(carry_cur_ptr + off, _floordiv(ws, P)[:, None] * RING + r[None, :], mask=m2)

    # store destination: the completed block's arithmetic row, else the row's scratch row
    fl = tl.load(snap_ptr + row * snap_stride + pos, mask=mb & (pos < snap_w), other=-1)
    rob = _floordiv(fl, RATIO)
    tl.store(cmp_dst_attn_ptr + b, tl.where(should, rob, row + CMP_BASE), mask=mb)
    if HAS_IDX:
        tl.store(cmp_dst_idx_ptr + b, tl.where(should, rob, row + IDX_BASE), mask=mb)


class RatioIndexCtx:
    """One decode step's derived indices for every layer of a single compress ratio.

    Layers of the same ratio derive IDENTICAL values (the inputs -- ``pos``, the
    snapshot, the window slots -- are layer-invariant and the scratch bases are a
    function of the ratio), so this is computed once per step per ratio class and
    handed to all of them.
    """

    __slots__ = (
        "ratio", "ring_size", "idx_mod", "should", "should3", "ovl_slot",
        "carry_prev", "carry_cur", "freq_idx", "valid", "cmp_counts",
        "cmp_dst_attn", "cmp_dst_idx", "topk_idxs",
    )

    def cmp_dst(self, tier: str) -> torch.Tensor:
        return self.cmp_dst_attn if tier == "attn" else self.cmp_dst_idx


def decode_index_ctx(
    pos, rows, window_slots, prev_window_slots, snap, *,
    ratio: int, ring_size: int, P: int, cap: int,
    cmp_base: int, idx_base: int | None, overlap: bool,
) -> RatioIndexCtx:
    B = pos.shape[0]
    dev = pos.device
    i64 = dict(dtype=torch.int64, device=dev)
    ctx = RatioIndexCtx()
    ctx.ratio = ratio
    ctx.ring_size = ring_size
    ctx.idx_mod = torch.empty(B, **i64)
    should = torch.empty(B, dtype=torch.bool, device=dev)
    ctx.should = should
    ctx.should3 = should.view(B, 1, 1)
    ctx.ovl_slot = torch.empty(B, **i64) if overlap else None
    ctx.carry_prev = torch.empty(B, ring_size, **i64)
    ctx.carry_cur = torch.empty(B, ring_size, **i64)
    ctx.freq_idx = torch.empty(B, **i64)
    ctx.valid = torch.empty(B, **i64)
    cmp_counts = torch.empty(B, dtype=torch.int32, device=dev)
    ctx.cmp_counts = cmp_counts.view(B, 1)
    ctx.cmp_dst_attn = torch.empty(B, **i64)
    ctx.cmp_dst_idx = torch.empty(B, **i64) if idx_base is not None else None
    ctx.topk_idxs = None
    _decode_index_ctx_kernel[(1,)](
        pos, rows, window_slots, prev_window_slots, snap,
        ctx.idx_mod, should, ctx.ovl_slot if overlap else ctx.idx_mod,
        ctx.carry_prev, ctx.carry_cur,
        ctx.freq_idx, ctx.valid, cmp_counts,
        ctx.cmp_dst_attn, ctx.cmp_dst_idx if idx_base is not None else ctx.cmp_dst_attn,
        B, snap.stride(0), snap.shape[1],
        RATIO=ratio, RING=ring_size, P=P, CAP=cap,
        CMP_BASE=cmp_base, IDX_BASE=idx_base or 0,
        HAS_IDX=idx_base is not None, OVERLAP=overlap,
        BLOCK_B=_pow2(B), BLOCK_R=_pow2(ring_size),
    )
    return ctx


# --------------------------------------------------------------------------- #
# 3. compressed picks -> global rows, concatenated onto the window half
# --------------------------------------------------------------------------- #
@triton.jit
def _cmp_topk_to_global_kernel(
    picks_ptr, valid_ptr, rows_ptr, snap_ptr, wtopk_ptr, out_ptr,
    B, K, snap_stride, snap_w,
    WIN: tl.constexpr, RATIO: tl.constexpr, OFFSET: tl.constexpr,
    IDENTITY: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= B:
        return
    row = tl.load(rows_ptr + b)
    v = tl.load(valid_ptr + b)
    out_base = out_ptr + b * (WIN + K)

    # window half is copied through verbatim (the kernel masks -1)
    j = tl.arange(0, BLOCK_W)
    mw = j < WIN
    w = tl.load(wtopk_ptr + b * WIN + j, mask=mw, other=0)
    tl.store(out_base + j, w.to(tl.int32), mask=mw)

    # compressed half: pick -> block (or -1) -> full loc -> global cmp row
    k = tl.arange(0, BLOCK_K)
    mk = k < K
    if IDENTITY:
        pick = k.to(tl.int64)
    else:
        pick = tl.load(picks_ptr + b * K + k, mask=mk, other=0)
    blocks = tl.where(pick >= v, -1, pick + OFFSET)
    col = tl.where(blocks < 0, 0, blocks) * RATIO
    fl = tl.load(snap_ptr + row * snap_stride + col, mask=mk & (col < snap_w), other=-1)
    res = tl.where(blocks < 0, -1, _floordiv(fl, RATIO))
    tl.store(out_base + WIN + k, res.to(tl.int32), mask=mk)


def cmp_topk_to_global(
    picks, valid, rows, snap, window_topk, *, ratio: int, offset: int = 0,
    identity_k: int | None = None,
):
    """``[window | compressed]`` global slots as ONE int32 tensor ``[B, 1, win + K]``.

    ``picks`` is the indexer's raw top-k block indices ``[B, 1, K]``; pass
    ``identity_k=K`` instead (with ``picks=None``) for the positional selection the
    indexer-less ratio-128 layers use, where pick ``k`` IS block ``k``.
    """
    B = window_topk.shape[0]
    win = window_topk.shape[-1]
    K = identity_k if picks is None else picks.shape[-1]
    out = torch.empty(B, 1, win + K, dtype=torch.int32, device=window_topk.device)
    _cmp_topk_to_global_kernel[(B,)](
        picks if picks is not None else out, valid, rows, snap, window_topk, out,
        B, K, snap.stride(0), snap.shape[1],
        WIN=win, RATIO=ratio, OFFSET=offset,
        IDENTITY=picks is None, BLOCK_K=_pow2(K), BLOCK_W=_pow2(win),
    )
    return out


__all__ = [
    "RatioIndexCtx",
    "cmp_topk_to_global",
    "decode_index_ctx",
    "window_ring_ctx",
]
