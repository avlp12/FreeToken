"""DSV4 MoE router tail as one Triton launch (decode fixed-cost fusion).

The DeepSeek-V4 router is a GEMV followed by a chain of nine tiny elementwise /
selection ops, none of which moves more than ``n_routed_experts`` floats:

    scores -> softplus -> sqrt -> (+ e-score bias) -> topk -> gather
           -> sum -> divide -> * route_scale

At decode (one row, 256 experts) every one of those is launch-bound, and the
chain runs 86 times per token: once per layer for the real routing decision and
once more for the prefetcher's L+1 lookahead replay (43 layers each). Measured
on this box: 9 post-GEMV launches per scoring gate, 7 per hash gate, 891 launches
per token for the whole router bucket -- 0.91 ms at the 1.024 us per-node CUDA
graph dispatch floor, before any of the kernels do arithmetic.

``fused_router`` collapses that whole tail into one kernel: one program per token
row, one block covering all ``n_routed_experts`` scores, top-k by ``top_k``
iterated arg-max passes over registers, renorm and scale in the same program.
The GEMV stays separate on purpose -- it is the only part of the router that is
bandwidth-bound (256x4096 bf16), and it wants the whole GPU (128 CTAs at
``BLOCK_N=2``), not the one CTA a per-row router program gets.

Numerics: every output is BIT-IDENTICAL (``torch.equal``) to the torch chain --
weights, expert ids and pre-renorm selection scores alike; see
tests/dsv4/test_dsv4_router_fusion.py. That is not free, and it is the reason
this file reaches for libdevice instead of the plain Triton operators. Under
Triton's fast-math default ``tl.exp`` becomes ``ex2.approx``, ``tl.sqrt``
becomes ``sqrt.approx.f32`` and ``/`` becomes ``div.full.f32``, each off by up
to 1 ULP from the ATen kernel it is standing in for. A 1 ULP drift in the
*scores* is not a rounding detail here: it can reorder a near-tie in the top-k
and route the token to a different expert. The three exactness-critical
substitutions are ``libdevice.exp`` / ``libdevice.log1p`` (softplus),
``libdevice.sqrt_rn`` and ``libdevice.div_rn``; the renorm sum reproduces ATen's
four-accumulator reduce association (see the RENORM block). Ties in the top-k
break to the lowest expert id via ``tl.argmax(tie_break_left=True)``.

``FREETOKEN_UNFUSED_ROUTER=1`` restores the pre-fusion torch composition (see
``freetoken.models.deepseek_v4.moe.Gate``).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def unfused_router() -> bool:
    """``FREETOKEN_UNFUSED_ROUTER=1`` -> take the pre-fusion torch chain.

    The fused kernel is index-bit-identical by construction and tested as such,
    so it is the default; the old path stays reachable so a routing regression
    can be bisected against it without a revert.
    """
    return os.environ.get("FREETOKEN_UNFUSED_ROUTER", "0") not in ("0", "")


@triton.jit
def _sqrtsoftplus(s):
    """``F.softplus(s).sqrt()`` with ATen's exact device math.

    ATen's ``softplus_kernel`` is ``x > threshold ? x : log1p(exp(x))`` with
    beta=1, threshold=20 -- ``std::exp``/``std::log1p`` on float lower to
    ``__nv_expf``/``__nv_log1pf``, so the libdevice calls below are the same
    instructions, not an approximation of them. ``sqrt_kernel`` is ``::sqrt``,
    i.e. IEEE round-to-nearest -- ``sqrt_rn`` here.

    The three substitutions matter: measured over 2.36M inputs, ``tl.sqrt`` and
    ``libdevice.sqrt`` both lower to ``sqrt.approx.f32`` under Triton's
    fast-math default and disagree with ATen on 17% of them; ``tl.exp`` lowers
    to ``ex2.approx`` and disagrees on 37%; ``log(1+exp(x))`` instead of
    ``log1p(exp(x))`` disagrees on 42%. With ``libdevice.exp`` / ``libdevice.log1p``
    / ``libdevice.sqrt_rn`` the activation is bit-identical to
    ``F.softplus(s).sqrt()`` everywhere tested.
    """
    sp = tl.where(s > 20.0, s, libdevice.log1p(libdevice.exp(s)))
    return libdevice.sqrt_rn(sp)


@triton.jit
def _router_kernel(
    scores_ptr,           # fp32 [M, N] raw router logits (GEMV output)
    bias_ptr,             # fp32 [N] e-score selection bias, or unused
    tid_ptr,              # int64 [V, K] token-id -> expert table, or unused
    ids_ptr,              # int64 [M] flat token ids, or unused
    w_ptr,                # fp32 [M, K] out: renormalised, scaled weights
    i64_ptr,              # int64 [M, K] out: expert ids
    i32_ptr,              # int32 [M, K] out: same ids (skips a caller-side cast)
    sel_ptr,              # fp32 [M, K] out: pre-renorm gathered scores
    M, N,
    route_scale,
    TOPK: tl.constexpr,
    TK: tl.constexpr,      # next_pow2(TOPK)
    BLOCK_N: tl.constexpr,  # next_pow2(N)
    HASH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORM: tl.constexpr,
    WANT_I32: tl.constexpr,
    WANT_SEL: tl.constexpr,
):
    m = tl.program_id(0)
    pos = tl.arange(0, TK)
    out_off = m * TOPK + pos
    out_mask = pos < TOPK

    if HASH:
        # Hash routing: the expert set is a pure function of the token id, so no
        # scoring pass -- just the table row, then the gathered scores.
        tok = tl.load(ids_ptr + m)
        idx = tl.load(tid_ptr + tok * TOPK + pos, mask=out_mask, other=0)
        raw = tl.load(scores_ptr + m * N + idx, mask=out_mask, other=0.0)
        sel = _sqrtsoftplus(raw)
    else:
        offs = tl.arange(0, BLOCK_N)
        n_mask = offs < N
        raw = tl.load(scores_ptr + m * N + offs, mask=n_mask, other=0.0)
        act = _sqrtsoftplus(raw)
        # Selection runs on the biased scores; the *weights* come from the
        # unbiased ones (`act`) -- the e-score bias steers load balance only.
        if HAS_BIAS:
            biased = act + tl.load(bias_ptr + offs, mask=n_mask, other=0.0)
        else:
            biased = act
        biased = tl.where(n_mask, biased, float("-inf"))

        idx = tl.zeros((TK,), tl.int64)
        sel = tl.zeros((TK,), tl.float32)
        live = biased
        for j in tl.static_range(TOPK):
            # tie_break_left mirrors "lowest expert id wins", the only tie rule a
            # deterministic router can offer; exact ties in fp32 softplus+bias do
            # not occur for real activations.
            k = tl.argmax(live, axis=0, tie_break_left=True)
            # Exactly one lane survives the mask, so the reduce returns that lane's
            # value bit-exactly (x + 0.0 == x for the non-negative sqrt outputs).
            v = tl.sum(tl.where(offs == k, act, 0.0), axis=0)
            idx = tl.where(pos == j, k.to(tl.int64), idx)
            sel = tl.where(pos == j, v, sel)
            live = tl.where(offs == k, float("-inf"), live)

    if RENORM:
        # Reproduce ATen's association for `sel.sum(dim=-1)` exactly, so `weights`
        # is bit-identical and not merely close. `reduce_kernel` reduces a
        # contiguous fp32 last dim with `input_vec_size = 4`: four independent
        # accumulators, c[g] taking elements g, g+4, g+8, ... in increasing order,
        # then a stride-halving tree combine -- c0 += c2; c1 += c3; c0 += c1, i.e.
        # (c0 + c2) + (c1 + c3). Verified against torch for M in
        # {1,2,3,5,8,64,1024,8192} at top_k=6 (a plain left-to-right sum misses on
        # ~37% of rows); the bit-identity test is what pins this to the ATen build.
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        for j in tl.static_range(TOPK):
            # Exactly one lane of `sel` survives, so this reduce returns it exactly.
            v = tl.sum(tl.where(pos == j, sel, 0.0), axis=0)
            if j % 4 == 0:
                c0 = c0 + v
            elif j % 4 == 1:
                c1 = c1 + v
            elif j % 4 == 2:
                c2 = c2 + v
            else:
                c3 = c3 + v
        # `div_rn`, not `/`: Triton's fp32 divide lowers to `div.full.f32` under its
        # fast-math default, which is off by up to 1 ULP from ATen's `div.rn.f32`.
        w = libdevice.div_rn(sel, (c0 + c2) + (c1 + c3))
    else:
        w = sel
    w = w * route_scale

    tl.store(w_ptr + out_off, w, mask=out_mask)
    tl.store(i64_ptr + out_off, idx, mask=out_mask)
    if WANT_I32:
        tl.store(i32_ptr + out_off, idx.to(tl.int32), mask=out_mask)
    if WANT_SEL:
        tl.store(sel_ptr + out_off, sel, mask=out_mask)


def fused_router(
    scores: torch.Tensor,
    bias: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    *,
    top_k: int,
    route_scale: float,
    renormalize: bool = True,
    want_int32: bool = False,
    want_sel: bool = False,
):
    """One-launch router tail over raw fp32 router logits.

    ``scores``: ``[M, N]`` fp32 (the ``bf16_linear_fp32`` GEMV output).
    Returns ``(weights, indices_i64, indices_i32 | None, sel_scores | None)``.

    Every output is a fresh fixed-shape allocation and the kernel takes no host
    branch on device data, so the call is CUDA-graph capturable; the returned
    ``indices_i32`` is exclusively owned by the caller and therefore safe for the
    offload cache's in-place expert-id -> slot-id rewrite.
    """
    assert scores.dtype == torch.float32 and scores.is_cuda
    scores = scores.contiguous()
    M, N = scores.shape
    hash_route = tid2eid is not None
    if hash_route:
        assert input_ids is not None
        assert tid2eid.shape[1] == top_k and tid2eid.dtype == torch.int64
        assert tid2eid.is_contiguous()

    dev = scores.device
    weights = torch.empty((M, top_k), dtype=torch.float32, device=dev)
    idx64 = torch.empty((M, top_k), dtype=torch.int64, device=dev)
    idx32 = torch.empty((M, top_k), dtype=torch.int32, device=dev) if want_int32 else weights
    sel = torch.empty((M, top_k), dtype=torch.float32, device=dev) if want_sel else weights

    _router_kernel[(M,)](
        scores,
        bias if bias is not None else scores,
        tid2eid if hash_route else scores,
        input_ids.reshape(-1) if hash_route else scores,
        weights, idx64, idx32, sel,
        M, N,
        float(route_scale),
        TOPK=top_k,
        TK=triton.next_power_of_2(top_k),
        BLOCK_N=triton.next_power_of_2(N),
        HASH=hash_route,
        HAS_BIAS=bias is not None,
        RENORM=renormalize,
        WANT_I32=want_int32,
        WANT_SEL=want_sel,
        num_warps=4,
    )
    return (
        weights,
        idx64,
        idx32 if want_int32 else None,
        sel if want_sel else None,
    )


__all__ = ["fused_router", "unfused_router"]
