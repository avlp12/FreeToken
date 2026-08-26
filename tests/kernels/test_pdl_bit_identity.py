"""PDL enrollment gate: every enrolled kernel must be BIT-IDENTICAL with PDL on vs off.

PDL changes only *when* a kernel starts, never what it computes, so any difference here is
a real race -- a `gdc_wait()` placed after a load it was supposed to guard, or a
`gdc_launch_dependents()` placed before a store a successor consumes. Such races are
timing-dependent and silent, so each case is checked:

  * on the EAGER path and inside a CAPTURED CUDA GRAPH (the graph is where the programmatic
    edges actually form -- `cudaGraphGetEdges` reports the node dependency as
    `cudaGraphDependencyTypeProgrammatic` only for captured PDL launches);
  * over >= 20 replays with the *input mutated between replays*, so a consumer that read
    stale predecessor data would produce the previous replay's answer and be caught;
  * with a real in-graph PRODUCER in front of the enrolled kernel, so the barrier is
    exercised against genuine cross-kernel data flow rather than a quiescent buffer.

Run: FREETOKEN_PDL is toggled per-arm inside the process via `pdl_enabled.cache_clear()`;
Triton specializes on the ENABLE_PDL constexpr, so both variants get their own binary.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

from freetoken.kernel.triton.pdl import (  # noqa: E402
    LEVEL_GEMV, PDL_ENV, pdl_enabled, pdl_level,
)
from freetoken.utils.arch import is_sm90_supported  # noqa: E402

REPS = 25


def _set_pdl(on: bool) -> None:
    """Toggle to level 2 (everything enrolled) so the gate covers every call site."""
    os.environ[PDL_ENV] = str(LEVEL_GEMV) if on else "0"
    pdl_level.cache_clear()
    assert pdl_enabled(LEVEL_GEMV) is (on and is_sm90_supported())


def _fp8_weight(n: int, k: int, seed: int):
    """A block-scaled FP8 weight (N,K) + its e8m0 (N//128, K//128) scale codes."""
    from freetoken.kernel.triton.e4m3_compat import e4m3_act_dtype, e4m3_kernel_view

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.1).to(e4m3_act_dtype())
    sb = torch.full((n // 128, k // 128), 127, dtype=torch.uint8, device="cuda")
    return e4m3_kernel_view(w), sb


def _ab(chain, x_shape, dtype=torch.bfloat16, reps=REPS):
    """Run `chain(x) -> tensor` with PDL off then on; return the two lists of outputs.

    `chain` is called with a STATIC input buffer; the same deterministic sequence of input
    values is pushed through both arms, eagerly and then under graph capture.
    """
    feeds = [
        torch.randn(x_shape, generator=torch.Generator(device="cuda").manual_seed(1000 + i),
                    device="cuda", dtype=dtype)
        for i in range(reps)
    ]
    eager, graphed = {}, {}
    for on in (False, True):
        _set_pdl(on)
        x = torch.zeros(x_shape, device="cuda", dtype=dtype)

        # --- eager arm ---
        got = []
        for f in feeds:
            x.copy_(f)
            got.append(chain(x).clone())
        torch.cuda.synchronize()
        eager[on] = got

        # --- captured-graph arm ---
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                chain(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = chain(x)
        got = []
        for f in feeds:
            x.copy_(f)
            g.replay()
            torch.cuda.synchronize()
            got.append(out.clone())
        graphed[on] = got
    _set_pdl(False)
    return eager, graphed


def _assert_identical(name, eager, graphed):
    for i in range(len(eager[False])):
        assert torch.equal(eager[False][i], eager[True][i]), f"{name}: eager replay {i}"
        assert torch.equal(graphed[False][i], graphed[True][i]), f"{name}: graph replay {i}"
    # A chain whose output never varies with the input would pass vacuously.
    assert not torch.equal(graphed[False][0], graphed[False][1]), \
        f"{name}: output is input-independent, the test proves nothing"


# ---------------------------------------------------------------------------------------
# Enrolled-kernel cases. Each chain puts a real producer in front of the enrolled kernel.
# ---------------------------------------------------------------------------------------

def test_fused_fp8_gemv_after_rmsnorm():
    """rms_norm (enrolled) -> fused_fp8_gemv (enrolled): an enrolled->enrolled edge, which
    is where both markers of both kernels are simultaneously live."""
    from freetoken.kernel.triton.dsv4.fp8_linear import fused_fp8_gemv
    from freetoken.kernel.triton.dsv4.norm import rms_norm

    n, k = 2048, 4096
    w, sb = _fp8_weight(n, k, seed=7)
    gain = torch.randn(k, device="cuda", dtype=torch.bfloat16)

    def chain(x):
        h = rms_norm(x, gain, 1e-6)
        return fused_fp8_gemv(h, w, sb, torch.bfloat16)

    _assert_identical("fused_fp8_gemv", *_ab(chain, (1, k)))


def test_fused_fp8_gemv_splitk_lock_epilogue():
    """SPLIT_K>1 forces the _EP_LOCK epilogue, where the trigger must come after the
    LAST-arriving program's reduce-and-store, not after its partial store."""
    from freetoken.kernel.triton.dsv4.fp8_linear import fused_fp8_gemv

    n, k = 1024, 4096          # table entry (4, 8, 1, 1) -> SPLIT_K == 8
    w, sb = _fp8_weight(n, k, seed=11)

    def chain(x):
        return fused_fp8_gemv(x * 1.5, w, sb, torch.bfloat16)

    _assert_identical("fused_fp8_gemv/_EP_LOCK", *_ab(chain, (1, k)))


def test_unfused_act_quant_gemv_reduce_chain():
    """act_quant (enrolled) -> splitk gemv (enrolled) -> splitk reduce (enrolled):
    three enrolled kernels back to back, the deepest programmatic chain we build."""
    from freetoken.kernel.triton.dsv4.fp8_linear import _fp8_act_gemv, act_quant_fp8

    n, k = 2048, 4096
    w, sb = _fp8_weight(n, k, seed=13)

    def chain(x):
        a, sa = act_quant_fp8(x, 128)
        return _fp8_act_gemv(a.reshape(k), sa.reshape(-1), w, sb, torch.bfloat16)

    _assert_identical("act_quant->gemv->reduce", *_ab(chain, (1, k)))


def test_bf16_gemv_fp32_after_rmsnorm():
    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
    from freetoken.kernel.triton.dsv4.norm import rms_norm

    n, k = 1024, 2048
    w = (torch.randn(n, k, generator=torch.Generator(device="cuda").manual_seed(3),
                     device="cuda") * 0.05).to(torch.bfloat16)
    gain = torch.randn(k, device="cuda", dtype=torch.bfloat16)

    def chain(x):
        return bf16_linear_fp32(rms_norm(x, gain, 1e-6), w)

    _assert_identical("bf16_gemv_fp32", *_ab(chain, (1, k)))


def test_inv_rms_after_producer():
    from freetoken.kernel.triton.dsv4.norm import inv_rms

    def chain(x):
        return inv_rms(x * 2.0, 1e-6)

    _assert_identical("inv_rms", *_ab(chain, (4, 4096)))


def test_swiglu_after_two_gemvs():
    """Two producers feed one enrolled consumer -- the trigger of BOTH must have fired
    before swiglu's gdc_wait clears."""
    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
    from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu

    n, k = 1024, 2048
    g = torch.Generator(device="cuda").manual_seed(5)
    w1 = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    w3 = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.bfloat16)

    def chain(x):
        gate = bf16_linear_fp32(x, w1).to(torch.bfloat16)
        up = bf16_linear_fp32(x, w3).to(torch.bfloat16)
        return fused_swiglu(gate, up, 7.0, torch.bfloat16)

    _assert_identical("swiglu", *_ab(chain, (1, k)))


def test_gated_pool_after_producer():
    from freetoken.kernel.triton.dsv4.compress import gated_pool

    B, R, D = 4, 8, 512

    def chain(x):
        kv = x.reshape(B, R, D)
        return gated_pool(kv, kv * 0.5, torch.bfloat16)

    _assert_identical("gated_pool", *_ab(chain, (B * R, D)))


def test_rmsnorm_standalone():
    from freetoken.kernel.triton.dsv4.norm import rms_norm

    gain = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    def chain(x):
        return rms_norm(x * 1.25, gain, 1e-6)

    _assert_identical("rms_norm", *_ab(chain, (8, 4096)))


# ---------------------------------------------------------------------------------------
# hc.py / hc_fused.py / sinkhorn.py / rope.py / norm_rope.py / router.py / sparse_attn.py /
# offload_kernels.py enrollments.
# ---------------------------------------------------------------------------------------

def test_hc_pre_combine_after_producer():
    from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

    M, HC, D = 4, 4, 256

    def chain(x):
        xf = (x.reshape(M, HC, D) * 1.5).float()  # real producer: bf16*const -> fp32
        pre = xf[:, :, 0].sigmoid()
        return hc_pre_combine(xf, pre, torch.bfloat16)

    _assert_identical("hc_pre_combine", *_ab(chain, (M, HC * D)))


def test_hc_post_combine_after_producer():
    from freetoken.kernel.triton.dsv4.hc import hc_post_combine

    M, HC, D = 4, 4, 256

    def chain(x):
        r = x.reshape(M, HC, D)
        a = (r[:, 0] * 1.5).contiguous()             # [M, D] bf16
        residual = (r * 0.5).contiguous()            # [M, HC, D] bf16
        post = residual[:, :, 0].float().sigmoid()    # [M, HC] fp32
        comb = residual[:, :, :HC].float() * 0.1      # [M, HC, HC] fp32
        return hc_post_combine(a, residual, post, comb)

    _assert_identical("hc_post_combine", *_ab(chain, (M, HC * D)))


def test_hc_stage_after_producer():
    """SPLITK == 2 at this shape, so the epilogue trigger's placement after the `if last`
    reduction race (mirroring fp8_linear's _EP_LOCK epilogue) is actually exercised."""
    from freetoken.kernel.triton.dsv4.hc_fused import hc_stage

    hc_mult, dim, tokens = 2, 128, 1
    hc_dim = hc_mult * dim
    mixn = (2 + hc_mult) * hc_mult
    g = torch.Generator(device="cuda").manual_seed(21)
    hc_fn = (torch.randn(mixn, hc_dim, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    hc_scale = torch.rand(3, device="cuda", dtype=torch.float32) + 0.5
    hc_base = torch.randn(mixn, device="cuda", dtype=torch.float32) * 0.1

    def chain(x):
        stream = (x.reshape(tokens, hc_dim) * 1.5).to(torch.bfloat16)
        _, y, _, _ = hc_stage(
            stream, None, hc_fn, hc_scale, hc_base,
            hc_mult=hc_mult, sinkhorn_iters=2, hc_eps=1e-6, norm_eps=1e-6,
            tokens=tokens, dim=dim,
        )
        return y

    _assert_identical("hc_stage", *_ab(chain, (tokens, hc_dim)))


def test_hc_split_sinkhorn_after_producer():
    from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

    n, hc_mult = 4, 4
    mixn = (2 + hc_mult) * hc_mult
    hc_scale = torch.rand(3, device="cuda", dtype=torch.float32) + 0.5
    hc_base = torch.randn(mixn, device="cuda", dtype=torch.float32) * 0.1

    def chain(x):
        mixes = (x.reshape(n, mixn) * 2.0).float()
        _, _, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, 4, 1e-6)
        return comb

    _assert_identical("hc_split_sinkhorn", *_ab(chain, (n, mixn)))


def test_rope_decode_after_producer():
    from freetoken.kernel.triton.dsv4.rope import rope_decode_inplace

    B, H, rope_dim = 4, 2, 64
    freqs = torch.polar(
        torch.ones(8, rope_dim // 2, device="cuda"),
        torch.rand(8, rope_dim // 2, device="cuda"),
    )
    positions = torch.arange(B, device="cuda", dtype=torch.int64) % 8

    def chain(x):
        q = (x.reshape(B, 1, H, rope_dim) * 1.5).to(torch.bfloat16).contiguous()
        return rope_decode_inplace(q, freqs, inverse=False, positions=positions)

    _assert_identical("rope_decode_inplace", *_ab(chain, (B, H * rope_dim)))


def test_rms_norm_rope_decode_after_producer():
    from freetoken.kernel.triton.dsv4.norm_rope import rms_norm_rope_decode

    B, D, rope_dim = 4, 256, 64
    gain = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones(8, rope_dim // 2, device="cuda"),
        torch.rand(8, rope_dim // 2, device="cuda"),
    )
    positions = torch.arange(B, device="cuda", dtype=torch.int64) % 8

    def chain(x):
        h = (x.reshape(B, 1, D) * 1.5).to(torch.bfloat16)
        return rms_norm_rope_decode(h, gain, 1e-6, freqs, positions, rope_dim)

    _assert_identical("rms_norm_rope_decode", *_ab(chain, (B, D)))


def test_fused_router_after_producer():
    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
    from freetoken.kernel.triton.dsv4.router import fused_router

    K, N = 128, 32
    w = (torch.randn(N, K, generator=torch.Generator(device="cuda").manual_seed(17),
                     device="cuda") * 0.05).to(torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.01

    def chain(x):
        scores = bf16_linear_fp32(x, w)  # [1, N] fp32 -- enrolled->enrolled edge
        weights, _, _, _ = fused_router(scores, bias, None, None, top_k=4, route_scale=1.0)
        return weights

    _assert_identical("fused_router", *_ab(chain, (1, K)))


def test_sparse_attn_splitk_merge_after_producer():
    """Forces the split-K decode path (n_splits > 1) so the merge kernel's stage-1
    predecessor is exercised, not the single-program fallback."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    b, m, h, d = 1, 1, 16, 64
    n_window, n_cmp, topk = 8, 512, 512
    g = torch.Generator(device="cuda").manual_seed(31)
    window_pool = (torch.randn(64, d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    cmp_pool = (torch.randn(n_cmp, d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    attn_sink = torch.randn(h, device="cuda", dtype=torch.float32) * 0.1
    win_idx = torch.arange(n_window, device="cuda", dtype=torch.int32).view(1, 1, n_window)
    cmp_idx = (torch.arange(topk - n_window, device="cuda", dtype=torch.int32) % n_cmp).view(1, 1, -1)
    topk_idxs = torch.cat([win_idx, cmp_idx], dim=-1)

    def chain(x):
        q = (x.reshape(b, m, h, d) * 1.5).to(torch.bfloat16)
        return sparse_attn_paged(
            q, window_pool, cmp_pool, attn_sink, topk_idxs, n_window,
            softmax_scale=1.0 / (d ** 0.5),
        )

    _assert_identical("sparse_attn_splitk_merge", *_ab(chain, (b * m, h * d)))


def test_protect_slots_after_producer():
    """cache.usage is mutated in place, so `chain` zeroes it first -- otherwise state
    would leak across the PDL-off and PDL-on arms of `_ab`, which share this closure's
    `cache` object and would no longer be an apples-to-apples comparison."""
    from types import SimpleNamespace

    from freetoken.moe.offload_kernels import protect_slots

    cache_size = 64
    cache = SimpleNamespace(
        usage=torch.zeros(cache_size, device="cuda", dtype=torch.int64),
        step=torch.zeros((), device="cuda", dtype=torch.int64),
    )

    def chain(x):
        cache.usage.zero_()
        # Real producer: derive slot ids from x, as ensure_experts' slot rewrite would.
        slots = (x[:8].abs().long() % cache_size).contiguous()
        protect_slots(cache, slots)
        return cache.usage.clone()

    _assert_identical("protect_slots", *_ab(chain, (64,)))
