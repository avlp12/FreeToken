"""One-launch manifold-constrained Hyper-Connections (mHC) stage for DeepSeek-V4.

What this replaces
------------------
At every one of DSV4-Flash's 86 hyper-connection sites the reference composition is

    x_out = hc_post_combine(a, residual, post, comb)      # previous sublayer's re-expand
    xf    = x_out.float()                                  #  \
    rsqrt = inv_rms(x_out, norm_eps)                       #   |
    mixes = F.linear(xf, hc_fn) * rsqrt                    #   |  hc_pre
    pre, post, comb = hc_split_sinkhorn(mixes, ...)        #   |
    y     = hc_pre_combine(x_out, pre)                     #  /
    y     = rms_norm(y, norm_weight, norm_eps)             # the sublayer's input norm

which is 9 kernel launches (``F.linear`` at batch 1 is a two-stage cuBLAS gemv) for
~10 us of arithmetic. Under CUDA graphs the 1.024 us/node dispatch floor alone costs
more than the work. This module collapses the whole chain into ONE launch.

Structure (the same decomposition DeepSeek's TileKernels / SGLang ship as
``mhc_pre_gemm_sqrsum_splitk`` + ``mhc_pre_big_fuse``, but as a single Triton kernel):

* **split-K body**, ``SPLITK`` programs per token. Each owns a contiguous slice of the
  ``hc_mult*dim`` flat residual. It (a) re-expands its slice from the pending
  ``hc_post`` inputs when there is one, (b) accumulates that slice's partial
  ``hc_fn @ x`` for all ``(2+hc)*hc`` mix outputs and its partial ``sum(x*x)``.
  This is where the 1.5 MB of ``hc_fn`` is read, so it needs many SMs -- at batch 1 a
  single-CTA "big fuse" is ~20x slower (measured), which is exactly why the reference
  implementations split K here too.
* **epilogue**, run by whichever program arrives last (deterministic *ordering* of the
  reduction, not of the arrival): reduce the ``SPLITK`` partials in fixed index order,
  apply the RMS scale, run the 20-iteration Sinkhorn, collapse the streams with ``pre``,
  and apply the sublayer's RMSNorm. No spin-wait: every program stores its partials,
  fences with a release atomic, and only the winner continues.

Numerics
--------
Every *expression* is the reference's, in the reference's order -- the Sinkhorn's 20
iterations and its ``+eps`` placement, the ``pre``/``post`` sigmoid forms, the
``hc_post`` re-expand's accumulation, the stream collapse, the RMSNorm, and the
rounding of the collapsed stream through the activation dtype before the norm reads it
(``hc_pre_combine`` stores bf16 and ``rms_norm`` loads it back). Nothing is
approximated and no iteration count moves. What a single kernel cannot preserve is the
*rounding* of four things, all at the fp32 ulp level:

1. ``hc_fn @ x`` -- split-K partials, reduced in a fixed index order, versus cuBLAS's
   own two-stage split. cuBLAS's order is not reproducible across shapes either.
2. ``sum(x*x)`` -- a per-block tree reduce summed across the K slices, versus
   ``inv_rms``'s strided vector accumulator (already documented as not bit-identical to
   ATen, for exactly this reason).
3. the Sinkhorn's row and column sums -- 4-element reductions whose association order
   follows the tile layout, and the fused kernel cannot run at the standalone Sinkhorn
   kernel's ``num_warps=1``. Sinkhorn is a contraction toward the doubly-stochastic
   projection, so this perturbs the iterate, not the fixed point.
4. the ``hc_post`` re-expand -- same expression, same order, but the surrounding gemv's
   register pressure makes the compiler contract the multiply-adds differently than it
   does in ``hc.py``'s standalone kernel. Isolated, the loop is bit-exact; fused, ~1e-5
   of elements land one bf16 step away, all of them cancellation cases.

Measured against the reference composition: the bf16 activation is ~99.97% bit-
identical and never more than two bf16 steps away; ``post``/``comb`` agree to
``rtol=1e-5``. Against an *fp64* ground truth the fused mix is no less accurate than
cuBLAS's -- at most shapes it is roughly 2x better, because split-K sums shorter chains.
``tests/dsv4/test_hc_fused.py`` pins all of this, including a reduction-free input for
which the whole downstream chain must match bit for bit.

Set ``FREETOKEN_UNFUSED_HC=1`` to restore the reference composition (see
``models/deepseek_v4/model.py``); the numerics tests compare against exactly that.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.pdl import gdc_launch_dependents, gdc_wait, pdl_enabled

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# One token per program in the split-K body, so the whole 1.5 MB of ``hc_fn`` is read
# once PER TOKEN, where the reference's gemm reads it once per site and amortises it
# across the batch. Fusing buys ~8 launches x 1.024 us per site and costs
# (tokens-1) x ~1.05 us of extra weight traffic, so it stops paying somewhere in the
# teens -- measured crossover on a 5090 is between 16 tokens (1.9x faster) and 32
# (0.7x). Above the threshold the model hands the site back to the reference
# composition, which is both correct and the faster of the two there.
#
# Lifting this would mean tiling the split-K body over tokens as well (SGLang's
# token_block=32), which is the right move for large-batch decode and prefill but is
# not what the batch-1 decode tail needs.
FUSE_MAX_TOKENS = 16

# Tuned on an RTX 5090 (sm_120, 170 SMs) against the 43-layer chain in
# benchmarks/dsv4_hc_stage.py. The split-K body is a pure weight read -- 1.5 MB of
# hc_fn per site, no reuse, nothing to hide the latency behind -- so it wants programs,
# not work per program: one K-block each and enough of them to cover the machine is
# ~1.6x faster than the four-blocks-per-program rule the reference implementations use
# for their (larger, batched) tiles. Past 128 programs the epilogue, which is serial per
# token however finely K is split, is all that is left.
BLOCK_K = 128
NUM_WARPS = 4
NUM_STAGES = 2
_SK_CAP = 128
_SK_MIN_BLOCKS = 1


@triton.jit
def _hc_stage_kernel(
    # residual stream in: either ``x`` directly, or the pending hc_post operands
    x_ptr, a_ptr, res_ptr, ipost_ptr, icomb_ptr,
    # mixing parameters + the following sublayer's RMSNorm weight
    fn_ptr, scale_ptr, base_ptr, nw_ptr,
    # split-K workspace
    partm_ptr, parts_ptr, sem_ptr,
    # outputs
    xo_ptr, out_ptr, post_ptr, comb_ptr,
    D, HCD,
    s_xm, s_am, s_rm, s_ipm, s_icm, s_fn, s_xom, s_om, s_pom, s_cm,
    norm_eps, hc_eps,
    HC: tl.constexpr, MIXN: tl.constexpr, MIXP: tl.constexpr, SPLITK: tl.constexpr,
    BLK_K: tl.constexpr, BLK_Y: tl.constexpr, ITERS: tl.constexpr,
    HAS_POST: tl.constexpr, HAS_W: tl.constexpr, SINKHORN: tl.constexpr,
    OUT: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    k = tl.program_id(0)
    tok = tl.program_id(1)
    mrow = tl.arange(0, MIXP)
    mmask = mrow < MIXN
    chunk = HCD // SPLITK
    start = k * chunk

    acc = tl.zeros((MIXP,), dtype=tl.float32)
    sq = tl.zeros((), dtype=tl.float32)
    # x_ptr (or, under HAS_POST, a_ptr/res_ptr/ipost_ptr/icomb_ptr) is the residual stream
    # written by the previous hyper-connection site; fn_ptr is a constant mixing weight
    # loaded in the same loop, so one barrier before the loop covers both.
    if ENABLE_PDL:
        gdc_wait()
    for off in range(0, chunk, BLK_K):
        f = start + off + tl.arange(0, BLK_K)
        if HAS_POST:
            # The pending sublayer re-expand, in the same pass:
            #   stream[q,d] = post[q]*a[d] + sum_p comb[p,q]*res[p,d]
            # ``chunk``, ``BLK_K`` and ``D`` are all powers of two with BLK_K <= D, so a
            # block never straddles two streams and ``q`` is uniform across it. Same
            # expression, same accumulation order as ``hc.py``'s standalone kernel.
            q = (start + off) // D
            d = f - q * D
            xv = tl.load(ipost_ptr + tok * s_ipm + q).to(tl.float32) * tl.load(
                a_ptr + tok * s_am + d).to(tl.float32)
            for p in tl.static_range(HC):
                c = tl.load(icomb_ptr + tok * s_icm + p * HC + q).to(tl.float32)
                r = tl.load(res_ptr + tok * s_rm + p * D + d).to(tl.float32)
                xv += c * r
            tl.store(xo_ptr + tok * s_xom + f, xv.to(OUT))
            xv = xv.to(OUT).to(tl.float32)  # the collapse below reads the stored dtype
        else:
            xv = tl.load(x_ptr + tok * s_xm + f).to(tl.float32)
        sq += tl.sum(xv * xv, axis=0)
        wt = tl.load(fn_ptr + mrow[:, None] * s_fn + f[None, :], mask=mmask[:, None], other=0.0)
        acc += tl.sum(wt * xv[None, :], axis=1)

    base_m = partm_ptr + tok * SPLITK * MIXP
    tl.store(base_m + k * MIXP + mrow, acc, mask=mmask)
    tl.store(parts_ptr + tok * SPLITK + k, sq)

    # Orders this program's own stores (partials, and under HAS_POST its slice of the
    # re-expanded stream) against the epilogue's reads. Needed even at SPLITK == 1,
    # where the epilogue reads stream elements written by other threads of this program.
    tl.debug_barrier()
    last = True
    if SPLITK > 1:
        # Release the stores before the counter is published; the winner acquires them.
        # No program ever waits -- the losers simply exit.
        old = tl.atomic_add(sem_ptr + tok + tl.arange(0, 1), 1, sem="acq_rel", scope="gpu")
        last = tl.max(old, axis=0) == SPLITK - 1

    if last:
        if SPLITK > 1:
            tl.store(sem_ptr + tok, 0)  # leave it armed for the next graph replay
        kk = tl.arange(0, SPLITK)
        h = tl.arange(0, HC)
        # .cg keeps the cross-program reads off the (non-coherent) L1.
        sq_all = tl.sum(tl.load(parts_ptr + tok * SPLITK + kk, cache_modifier=".cg"), axis=0)
        rsq = tl.rsqrt(sq_all / HCD + norm_eps)

        sc0 = tl.load(scale_ptr + 0)
        pre_raw = tl.sum(
            tl.load(base_m + kk[:, None] * MIXP + h[None, :], cache_modifier=".cg"), axis=0)
        pre = tl.sigmoid(pre_raw * rsq * sc0 + tl.load(base_ptr + h)) + hc_eps

        if SINKHORN:
            sc1 = tl.load(scale_ptr + 1)
            sc2 = tl.load(scale_ptr + 2)
            post_raw = tl.sum(
                tl.load(base_m + kk[:, None] * MIXP + (HC + h)[None, :], cache_modifier=".cg"),
                axis=0)
            tl.store(post_ptr + tok * s_pom + h,
                     2.0 * tl.sigmoid(post_raw * rsq * sc1 + tl.load(base_ptr + HC + h)))

            idx = h[:, None] * HC + h[None, :]
            c = tl.sum(
                tl.load(base_m + kk[:, None, None] * MIXP + (2 * HC + idx)[None, :, :],
                        cache_modifier=".cg"),
                axis=0) * rsq * sc2 + tl.load(base_ptr + 2 * HC + idx)
            c = c - tl.max(c, axis=1)[:, None]
            c = tl.exp(c)
            c = c / tl.sum(c, axis=1)[:, None]
            c = c + hc_eps
            c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
            for _ in range(ITERS - 1):
                c = c / (tl.sum(c, axis=1)[:, None] + hc_eps)
                c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
            tl.store(comb_ptr + tok * s_cm + idx, c)

        dof = tl.arange(0, BLK_Y)
        dmask = dof < D
        y = tl.zeros((BLK_Y,), dtype=tl.float32)
        for hh in tl.static_range(HC):
            p = tl.sum(tl.where(h == hh, pre, 0.0), axis=0)
            if HAS_POST:
                xv = tl.load(xo_ptr + tok * s_xom + hh * D + dof, mask=dmask, other=0.0,
                             cache_modifier=".cg")
            else:
                xv = tl.load(x_ptr + tok * s_xm + hh * D + dof, mask=dmask, other=0.0)
            y += p * xv.to(tl.float32)
        y = y.to(OUT).to(tl.float32)  # hc_pre_combine stores the activation dtype
        if HAS_W:
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / D + norm_eps)
            y = y * tl.load(nw_ptr + dof, mask=dmask, other=0.0).to(tl.float32)
        tl.store(out_ptr + tok * s_om + dof, y.to(OUT), mask=dmask)
    # Every program reaches here -- the `if last` branch above fully rejoins, exactly like
    # fp8_linear.py's _EP_LOCK epilogue. Only the last-arriving program's stores are what a
    # successor actually reads, but triggering from the losers too is harmless: the
    # successor's gdc_wait() only clears once EVERY program of this grid has triggered (or
    # retired), so the effective release is still gated by the winner's store above.
    if ENABLE_PDL:
        gdc_launch_dependents()


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def split_k_for(tokens: int, hc_dim: int, device_index: int) -> int:
    """Power-of-two K split: fill the machine, but keep >= 4 K-blocks per program.

    Same shape of heuristic as SGLang's ``_compute_num_split_for_mhc_pre``. It depends
    only on ``(tokens, hc_dim, device)``, so a given shape always reduces in the same
    order -- reproducibility does not depend on occupancy or arrival order.
    """
    cap = min(_sm_count(device_index) // max(tokens, 1),
              hc_dim // (_SK_MIN_BLOCKS * BLOCK_K), _SK_CAP)
    s = 1
    while s * 2 <= cap and hc_dim % (s * 2) == 0 and (hc_dim // (s * 2)) % BLOCK_K == 0:
        s *= 2
    return s


_WORKSPACE: dict[tuple[int, int, int, int],
                 tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _workspace(tokens: int, splitk: int, mixp: int, device: torch.device):
    """Scratch for the split-K partials and the arrival counter.

    Shared by every hyper-connection site: the stages are stream-ordered, so a site can
    never observe another's partials, inside a captured graph or out of it. The counter
    is left at zero by the winning program, so a replay finds it armed. Allocation is
    lazy but the engine always runs one eager forward at each captured shape first
    (``engine/graph.py``), so nothing is ever allocated from a graph's private pool.
    """
    # ``mixp`` is part of the key: the head's collapse mixes ``hc_mult`` gates while a
    # block's mixes ``(2+hc_mult)*hc_mult``, and both can land on the same token count.
    key = (device.index, tokens, splitk, mixp)
    ws = _WORKSPACE.get(key)
    if ws is None:
        ws = (
            torch.empty(tokens, splitk, mixp, dtype=torch.float32, device=device),
            torch.empty(tokens, splitk, dtype=torch.float32, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
        )
        _WORKSPACE[key] = ws
    return ws


def _launch(
    *, x, pending, hc_fn, hc_scale, hc_base, norm_weight, hc_mult, sinkhorn_iters,
    hc_eps, norm_eps, sinkhorn, tokens, dim, out_dtype, device,
):
    hc_dim = hc_mult * dim
    mixn = (2 + hc_mult) * hc_mult if sinkhorn else hc_mult
    mixp = triton.next_power_of_2(mixn)
    splitk = split_k_for(tokens, hc_dim, device.index)
    partm, parts, sem = _workspace(tokens, splitk, mixp, device)

    out = torch.empty(tokens, dim, dtype=out_dtype, device=device)
    if sinkhorn:
        post = torch.empty(tokens, hc_mult, dtype=torch.float32, device=device)
        comb = torch.empty(tokens, hc_mult, hc_mult, dtype=torch.float32, device=device)
    else:
        post = comb = out  # unused; a valid pointer keeps the launcher happy

    if pending is None:
        a = res = ipost = icomb = x
        xo = x
    else:
        a, res, ipost, icomb = pending
        xo = torch.empty(tokens, hc_dim, dtype=out_dtype, device=device)
        x = xo

    pdl = pdl_enabled()
    _hc_stage_kernel[(splitk, tokens)](
        x, a, res, ipost, icomb,
        hc_fn, hc_scale, hc_base, norm_weight if norm_weight is not None else out,
        partm, parts, sem,
        xo, out, post, comb,
        dim, hc_dim,
        x.stride(0), a.stride(0), res.stride(0), ipost.stride(0), icomb.stride(0),
        hc_fn.stride(0), xo.stride(0), out.stride(0), post.stride(0), comb.stride(0),
        norm_eps, hc_eps,
        HC=hc_mult, MIXN=mixn, MIXP=mixp, SPLITK=splitk,
        BLK_K=BLOCK_K, BLK_Y=triton.next_power_of_2(dim), ITERS=sinkhorn_iters,
        HAS_POST=pending is not None, HAS_W=norm_weight is not None, SINKHORN=sinkhorn,
        OUT=_TL[out_dtype], ENABLE_PDL=pdl, launch_pdl=pdl,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return xo, out, post, comb


def can_fuse(tokens: int, dim: int, hc_mult: int) -> bool:
    """Shapes the single-launch stage is both correct and profitable for."""
    hc_dim = hc_mult * dim
    return (
        tokens <= FUSE_MAX_TOKENS
        and hc_mult & (hc_mult - 1) == 0
        and dim & (dim - 1) == 0
        and hc_dim % BLOCK_K == 0
        and dim >= BLOCK_K
    )


def hc_stage(
    x: torch.Tensor | None,
    pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None = None,
    tokens: int,
    dim: int,
):
    """One launch: pending ``hc_post`` re-expand -> mix gemv + sqrsum -> Sinkhorn ->
    stream collapse -> the sublayer's RMSNorm.

    Exactly one of ``x`` (an already-materialised ``[tokens, hc_mult*dim]`` stream) and
    ``pending`` (``(a, residual, post, comb)`` from the previous sublayer, whose
    re-expand this call absorbs) is given.

    Returns ``(stream, y, post, comb)``: ``stream`` is the ``[tokens, hc_mult*dim]``
    residual this site branched from (the caller's next ``residual``), ``y`` is the
    ``[tokens, dim]`` normalised sublayer input, and ``post``/``comb`` are this site's
    re-expand operands to hand to the next stage.
    """
    src = x if x is not None else pending[0]
    return _launch(
        x=x, pending=pending, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base,
        norm_weight=norm_weight, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps, norm_eps=norm_eps, sinkhorn=True, tokens=tokens, dim=dim,
        out_dtype=src.dtype, device=src.device,
    )


def hc_head_stage(
    x: torch.Tensor | None,
    pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None = None,
    tokens: int,
    dim: int,
) -> torch.Tensor:
    """The output head's collapse: same stage without the Sinkhorn half (the head mixes
    with ``hc_mult`` gates only), optionally absorbing the last block's ``hc_post`` and
    the final RMSNorm. Returns the ``[tokens, dim]`` collapsed stream."""
    src = x if x is not None else pending[0]
    _, y, _, _ = _launch(
        x=x, pending=pending, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base,
        norm_weight=norm_weight, hc_mult=hc_mult, sinkhorn_iters=1,
        hc_eps=hc_eps, norm_eps=norm_eps, sinkhorn=False, tokens=tokens, dim=dim,
        out_dtype=src.dtype, device=src.device,
    )
    return y


__all__ = ["hc_stage", "hc_head_stage", "can_fuse", "split_k_for", "FUSE_MAX_TOKENS"]
