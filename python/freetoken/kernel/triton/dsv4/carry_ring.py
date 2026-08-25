"""Fused compress-state ring traffic for a DSV4 decode step.

The compressor's rolling carry lives per window PAGE in a fp32 ring block. One decode
step reads the previous token's block, drops the new token into its ``pos % ratio``
slot, feeds the result to the gated pool, rolls the overlap half when the block
completed, and writes the block back at the current token's page. Composed in torch
that is a gather, two ``clone``s, two ``scatter_``s, two overlap ``cat``s, two
``where``s with their copy-backs, a ``cat`` and a masked ``index_put`` plus the
ring's two scratch-clear fills -- 15 launches for an overlap (ratio-4) compressor and
9 without, times the 62 compressors DSV4-Flash runs per token.

None of it is arithmetic: it is pure movement between the ring block, the two
register halves and the gated pool's operands. So it collapses to two launches:

* :func:`carry_load` gathers the previous block, substitutes the new token's row, and
  emits ``ks``/``ss`` (and, for an overlap compressor, the pool's ``[B, 2*ratio, d]``
  operands, which are just a different view of the same rows);
* :func:`carry_store` writes the advanced block back at the current page, applying
  the completion roll on the fly instead of mutating ``ks``/``ss`` in place.

Both take the ring rows PRE-RESOLVED by the step's fused index kernel, and both
replicate torch's addressing exactly -- negative row indices wrap to the buffer tail,
and the permanent scratch row (index ``-1``) is never written and always re-cleared,
which is what ``buffer[rows] = blocks`` followed by ``_clear_scratch()`` amounts to.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length() if n > 1 else 1


@triton.jit
def _carry_load_kernel(
    buf_ptr, rows_ptr, kv_ptr, score_ptr, slot_ptr,
    ks_ptr, ss_ptr, kve_ptr, sse_ptr,
    N_ROWS,
    R: tl.constexpr, ITEM: tl.constexpr, LD: tl.constexpr, D: tl.constexpr,
    RATIO: tl.constexpr, OVERLAP: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    r = tl.program_id(1)
    row = tl.load(rows_ptr + b * R + r)
    row = tl.where(row < 0, row + N_ROWS, row)  # torch's negative-index wrap
    is_new = r == tl.load(slot_ptr + b)

    offs = tl.arange(0, BLOCK)
    m = offs < ITEM
    ring_k = tl.load(buf_ptr + row * LD + offs, mask=m, other=0.0)
    ring_s = tl.load(buf_ptr + row * LD + ITEM + offs, mask=m, other=0.0)
    new_k = tl.load(kv_ptr + b * ITEM + offs, mask=m, other=0.0)
    new_s = tl.load(score_ptr + b * ITEM + offs, mask=m, other=0.0)
    k = tl.where(is_new, new_k, ring_k)
    s = tl.where(is_new, new_s, ring_s)
    dst = (b * R + r) * ITEM + offs
    tl.store(ks_ptr + dst, k, mask=m)
    tl.store(ss_ptr + dst, s, mask=m)

    if OVERLAP:
        # kv_eff = cat([ks[:, :ratio, :d], ks[:, ratio:, d:]], dim=1): row r takes the
        # A half below the split and the B half above it.
        shift = tl.where(r < RATIO, 0, D)
        od = tl.arange(0, BLOCK_D)
        md = od < D
        ring_ke = tl.load(buf_ptr + row * LD + shift + od, mask=md, other=0.0)
        ring_se = tl.load(buf_ptr + row * LD + ITEM + shift + od, mask=md, other=0.0)
        new_ke = tl.load(kv_ptr + b * ITEM + shift + od, mask=md, other=0.0)
        new_se = tl.load(score_ptr + b * ITEM + shift + od, mask=md, other=0.0)
        edst = (b * R + r) * D + od
        tl.store(kve_ptr + edst, tl.where(is_new, new_ke, ring_ke), mask=md)
        tl.store(sse_ptr + edst, tl.where(is_new, new_se, ring_se), mask=md)


def carry_load(buffer, ring_rows, kv, score, slot, *, item: int, ratio: int, d: int,
               overlap: bool):
    """``(ks, ss, kv_eff, score_eff)`` -- the previous page's carry block with this
    token's ``kv``/``score`` already in slot ``slot``.

    ``kv_eff``/``score_eff`` are ``None`` without overlap: the pool then consumes
    ``ks``/``ss`` directly (``item == d`` there).
    """
    B, R = ring_rows.shape
    dev = buffer.device
    ks = torch.empty(B, R, item, dtype=torch.float32, device=dev)
    ss = torch.empty(B, R, item, dtype=torch.float32, device=dev)
    kve = torch.empty(B, R, d, dtype=torch.float32, device=dev) if overlap else None
    sse = torch.empty(B, R, d, dtype=torch.float32, device=dev) if overlap else None
    _carry_load_kernel[(B, R)](
        buffer, ring_rows, kv, score, slot,
        ks, ss, kve if overlap else ks, sse if overlap else ss,
        buffer.shape[0],
        R=R, ITEM=item, LD=buffer.shape[1], D=d, RATIO=ratio, OVERLAP=overlap,
        BLOCK=_pow2(item), BLOCK_D=_pow2(d),
    )
    return ks, ss, kve, sse


@triton.jit
def _carry_store_kernel(
    buf_ptr, rows_ptr, ks_ptr, ss_ptr, should_ptr,
    N_ROWS, NDATA,
    R: tl.constexpr, ITEM: tl.constexpr, LD: tl.constexpr,
    RATIO: tl.constexpr, OVERLAP: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < ITEM
    if pid >= NDATA:
        # The ring's permanent scratch row (index -1), re-cleared after every write --
        # exactly what CompressStateRing._clear_scratch does. No data program can
        # target it (they are masked below), so there is nothing to race with.
        tl.store(buf_ptr + (N_ROWS - 1) * LD + offs, tl.zeros([BLOCK], tl.float32), mask=m)
        tl.store(
            buf_ptr + (N_ROWS - 1) * LD + ITEM + offs,
            tl.full([BLOCK], float("-inf"), tl.float32), mask=m,
        )
    else:
        b = pid // R
        r = pid % R
        row = tl.load(rows_ptr + pid)
        row = tl.where(row < 0, row + N_ROWS, row)  # torch's negative-index wrap
        src = r
        if OVERLAP:
            # ks[:, :ratio] = where(should, ks[:, ratio:], ks[:, :ratio]) -- the
            # completed block's B half becomes the next block's overlap seed.
            done = tl.load(should_ptr + b) != 0
            src = tl.where((r < RATIO) & done, r + RATIO, r)
        k = tl.load(ks_ptr + (b * R + src) * ITEM + offs, mask=m, other=0.0)
        s = tl.load(ss_ptr + (b * R + src) * ITEM + offs, mask=m, other=0.0)
        # A write landing on the scratch row would be undone by the re-clear anyway.
        w = m & (row != N_ROWS - 1)
        tl.store(buf_ptr + row * LD + offs, k, mask=w)
        tl.store(buf_ptr + row * LD + ITEM + offs, s, mask=w)


def carry_store(buffer, ring_rows, ks, ss, should, *, item: int, ratio: int, overlap: bool):
    """Write the advanced carry block back at ``ring_rows``, rolling the overlap half
    where the block completed, and re-clear the ring's scratch row."""
    B, R = ring_rows.shape
    _carry_store_kernel[(B * R + 1,)](
        buffer, ring_rows, ks, ss, should.view(torch.int8),
        buffer.shape[0], B * R,
        R=R, ITEM=item, LD=buffer.shape[1], RATIO=ratio, OVERLAP=overlap,
        BLOCK=_pow2(item),
    )


__all__ = ["carry_load", "carry_store"]
