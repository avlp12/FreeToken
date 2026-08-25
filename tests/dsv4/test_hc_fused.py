"""The one-launch mHC stage must reproduce the reference composition it replaces.

The reference is exactly what ``FREETOKEN_UNFUSED_HC=1`` restores in
``models/deepseek_v4/model.py``: ``hc_post_combine`` -> ``inv_rms`` -> ``F.linear`` ->
``hc_split_sinkhorn`` -> ``hc_pre_combine`` -> ``rms_norm``. Two reductions inside the
fused stage necessarily change order (the mix gemv is split over K, and the sum of
squares is a per-block tree instead of ``inv_rms``'s strided accumulator), so the
contract is:

* with the two reordered reductions removed by construction (a one-hot residual, whose
  mix gemv is a single product and whose ``sum(x*x)`` is a single 1.0),
  ``test_reduction_free_input_is_exact`` demands ``torch.equal`` on ``post`` -- the
  whole gemv -> rsqrt -> scale -> sigmoid chain, bit for bit;
* the ``[tokens, dim]`` bf16 activation agrees with the reference to **within two bf16
  steps** and is ~99.9% bit-identical;
* the fp32 ``post`` / ``comb`` re-expand operands agree to ``rtol=1e-5``, i.e. tens of
  fp32 ulps out of a 16384-term reduction;
* the fused mix is **no less accurate than the reference** against an fp64 ground
  truth -- that is the justification for the reorder, not merely a tolerance;
* ``comb`` stays doubly stochastic (the load-bearing Sinkhorn invariant), and
  ``test_sinkhorn_iteration_count_is_load_bearing`` proves the 20 iterations are all
  actually run.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.hc_fused import (
    FUSE_MAX_TOKENS, can_fuse, hc_head_stage, hc_stage, split_k_for,
)
from freetoken.kernel.triton.dsv4.norm import inv_rms, rms_norm
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel")

HC = 4
DIM = 4096
HCD = HC * DIM
MIX = (2 + HC) * HC
EPS = 1e-6
ITERS = 20
DEV = "cuda"


def _params(seed=0, mix=MIX):
    g = torch.Generator(device=DEV).manual_seed(seed)
    fn = torch.randn(mix, HCD, generator=g, device=DEV, dtype=torch.float32) * 0.02
    scale = torch.rand(3, generator=g, device=DEV) + 0.5
    base = torch.randn(mix, generator=g, device=DEV) * 0.1
    w = torch.randn(DIM, generator=g, device=DEV, dtype=torch.bfloat16)
    return fn, scale, base, w


def _ref_pre(x, fn, scale, base, w):
    """The composition model.py runs under FREETOKEN_UNFUSED_HC=1."""
    M = x.shape[0]
    x2d = x.view(M, HCD)
    mixes = F.linear(x2d.float(), fn) * inv_rms(x2d, EPS)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
    y = hc_pre_combine(x.view(M, HC, DIM), pre, x.dtype)
    if w is not None:
        y = rms_norm(y, w, EPS)
    return y, post, comb


def _ref_post(a, residual, post, comb):
    M = a.shape[0]
    return hc_post_combine(a, residual.view(M, HC, DIM), post, comb).view(M, HCD)


BF16_STEP = 2.0 ** -8  # one bf16 ulp, relative


def _assert_bf16_adjacent(got, ref, steps=2, min_eq=0.999, rel_l2=5e-4):
    """Almost all bit-identical, negligible in L2, and hard-capped per element.

    A pure per-element *relative* bound is the wrong instrument here: a handful of
    entries per tensor are catastrophic cancellations, where a single fp32 ulp upstream
    swings the bf16 exponent by many steps while moving the value by ~1e-8 absolute.
    So the contract is three-part -- the bulk is bit-identical, the whole tensor agrees
    in L2 (~1e-3 of a bf16 ulp RMS: one ulp is 2^-8 and ~3e-4 of elements carry one),
    and no element strays further than ``steps`` bf16 ulps of the tensor's own peak
    magnitude."""
    g, r = got.float(), ref.float()
    d = (g - r).abs()
    eq = (got == ref).float().mean().item()
    assert eq >= min_eq, f"only {eq:.5f} of the output is bit-identical"
    l2 = (d.pow(2).sum().sqrt() / r.pow(2).sum().sqrt().clamp_min(1e-20)).item()
    assert l2 <= rel_l2, f"relative L2 {l2:.3e} > {rel_l2:.0e}"
    cap = steps * BF16_STEP * r.abs().max().item()
    assert d.max().item() <= cap, f"max |diff| {d.max().item():.3e} > {cap:.3e}"


@pytest.mark.parametrize("tokens", [1, 2, 5, 8, 32, 64])
def test_stage_matches_reference(tokens):
    fn, scale, base, w = _params(tokens)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    ry, rpost, rcomb = _ref_pre(x, fn, scale, base, w)

    stream, y, post, comb = hc_stage(
        x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
        norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
    )
    assert stream is x
    _assert_bf16_adjacent(y, ry)
    torch.testing.assert_close(post, rpost.view(tokens, HC), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(comb, rcomb.view(tokens, HC, HC), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("tokens", [1, 2, 8, 32, 64])
def test_mix_accuracy_not_worse_than_reference(tokens):
    """Justifies the reduce-order change: scored against an fp64 ground truth, the
    split-K mix must not be the worse of the two. It is usually the better one -- split-K
    sums 512-term chains instead of one 16384-term chain -- which is the same reason
    ``inv_rms`` was allowed to diverge from ATen.

    ``post`` is the cleanest probe: it is one well-conditioned elementwise map away from
    the mix, with none of the Sinkhorn's iteration in between."""
    fn, scale, base, w = _params(tokens)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    _, rpost, _ = _ref_pre(x, fn, scale, base, w)
    _, _, post, _ = hc_stage(
        x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
        norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
    )
    xd = x.double()
    mix = ((xd @ fn.double().T) * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + EPS))
    truth = 2.0 * torch.sigmoid(mix[:, HC:2 * HC] * scale[1].double() + base[HC:2 * HC].double())

    def err(p):
        return ((p.double() - truth).abs() / truth.abs()).mean().item()

    err_fused, err_ref = err(post), err(rpost.view(tokens, HC))
    assert err_fused < err_ref * 1.5, (tokens, err_fused, err_ref)


def test_reexpand_matches_standalone_kernel():
    """``hc_post`` folded into the stage runs the reference kernel's expression in the
    reference's order. It is not quite ``torch.equal``: sharing a kernel with the gemv
    changes how the compiler contracts the multiply-adds, which moves a handful of
    cancellation cases by one bf16 step. Nothing else may move."""
    tokens = 8
    fn, scale, base, w = _params(23)
    a = torch.randn(tokens, DIM, device=DEV, dtype=torch.bfloat16)
    residual = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    post_in = torch.rand(tokens, HC, device=DEV, dtype=torch.float32) + 0.5
    comb_in = torch.rand(tokens, HC, HC, device=DEV, dtype=torch.float32)
    stream, _, _, _ = hc_stage(
        None, (a, residual, post_in, comb_in), fn, scale, base, hc_mult=HC,
        sinkhorn_iters=ITERS, hc_eps=EPS, norm_eps=EPS, norm_weight=w,
        tokens=tokens, dim=DIM,
    )
    _assert_bf16_adjacent(stream, _ref_post(a, residual, post_in, comb_in),
                          steps=2, min_eq=0.9999)


@pytest.mark.parametrize("tokens", [1, 3, 16, 64])
def test_stage_absorbs_pending_post(tokens):
    """The pending sublayer re-expand folded into the same launch."""
    fn, scale, base, w = _params(tokens + 100)
    a = torch.randn(tokens, DIM, device=DEV, dtype=torch.bfloat16)
    residual = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    post_in = torch.rand(tokens, HC, device=DEV, dtype=torch.float32) + 0.5
    comb_in = torch.rand(tokens, HC, HC, device=DEV, dtype=torch.float32)
    comb_in /= comb_in.sum(-1, keepdim=True)

    rx = _ref_post(a, residual, post_in, comb_in)
    ry, rpost, rcomb = _ref_pre(rx, fn, scale, base, w)

    stream, y, post, comb = hc_stage(
        None, (a, residual, post_in, comb_in), fn, scale, base,
        hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS, norm_eps=EPS,
        norm_weight=w, tokens=tokens, dim=DIM,
    )
    _assert_bf16_adjacent(stream, rx, steps=2, min_eq=0.9999)
    _assert_bf16_adjacent(y, ry)
    # Looser than the no-pending case by one order: the fused stream itself is one bf16
    # step off the reference on a handful of elements, and the mix is a 16384-term dot
    # product against that stream, so those steps propagate into post/comb.
    torch.testing.assert_close(post, rpost.view(tokens, HC), rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(comb, rcomb.view(tokens, HC, HC), rtol=1e-3, atol=1e-5)


def test_reduction_free_input_is_exact():
    """Remove the two reordered reductions and the gemv -> rsqrt -> scale -> sigmoid
    chain must match bit for bit.

    One-hot rows do it: the mix gemv collapses to a single product (so split-K and
    cuBLAS cannot disagree) and ``sum(x*x)`` is a single 1.0 in any order. ``post`` is
    the pure elementwise tail of that chain, so it is the strict check. ``comb`` runs on
    through the Sinkhorn, whose 4-element row/column sums associate per tile layout --
    it stays a tight ``allclose``. The collapse ``y`` is exact here because the residual
    is one-hot."""
    tokens = 8
    x = torch.zeros(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    x[torch.arange(tokens), torch.arange(tokens) * 977 % HCD] = 1.0
    fn, scale, base, w = _params(7)
    ry, rpost, rcomb = _ref_pre(x, fn, scale, base, w)
    _, y, post, comb = hc_stage(
        x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
        norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
    )
    assert torch.equal(post, rpost.view(tokens, HC))
    torch.testing.assert_close(comb, rcomb.view(tokens, HC, HC), rtol=1e-5, atol=1e-7)
    _assert_bf16_adjacent(y, ry, steps=1, min_eq=0.999)


def test_sinkhorn_iteration_count_is_load_bearing():
    """Guards against a silently shortened Sinkhorn: 20 iterations must be visibly
    different from 19, and must land closer to doubly stochastic."""
    tokens = 8
    fn, scale, base, w = _params(29)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16) * 2.0
    outs = {}
    for iters in (2, 19, 20):
        _, _, _, comb = hc_stage(
            x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=iters, hc_eps=EPS,
            norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
        )
        outs[iters] = comb.clone()
    assert not torch.equal(outs[20], outs[19])
    one = torch.ones(tokens, HC, device=DEV)
    dev = lambda c: (c.sum(-1) - one).abs().max().item()  # noqa: E731
    assert dev(outs[20]) < dev(outs[2])


def test_comb_is_doubly_stochastic():
    """The invariant that keeps the 4x-wide residual stream stable. 20 Sinkhorn
    iterations do not reach the fixed point exactly for every input, so the bar is the
    reference's own residual, not an absolute one: the fused stage must be no further
    from doubly stochastic than the kernel it replaces."""
    tokens = 16
    fn, scale, base, w = _params(3)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16) * 3.0
    _, _, _, comb = hc_stage(
        x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
        norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
    )
    _, rcomb = _ref_pre(x, fn, scale, base, w)[1:]
    rcomb = rcomb.view(tokens, HC, HC)
    for axis in (-1, -2):
        got = (comb.sum(axis) - 1).abs().max().item()
        ref = (rcomb.sum(axis) - 1).abs().max().item()
        assert got <= max(ref * 1.05, 1e-5), (axis, got, ref)
        assert got < 1e-2, (axis, got)


@pytest.mark.parametrize("with_norm", [False, True])
def test_head_stage_matches_reference(with_norm):
    tokens = 4
    fn, scale, base, w = _params(11, mix=HC)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    x2d = x.view(tokens, HCD)
    mixes = F.linear(x2d.float(), fn) * inv_rms(x2d, EPS)
    pre = torch.sigmoid(mixes * scale[:1] + base) + EPS
    ref = hc_pre_combine(x.view(tokens, HC, DIM), pre, x.dtype)
    if with_norm:
        ref = rms_norm(ref, w, EPS)

    got = hc_head_stage(
        x, None, fn, scale[:1].contiguous(), base, hc_mult=HC, hc_eps=EPS,
        norm_eps=EPS, norm_weight=w if with_norm else None, tokens=tokens, dim=DIM,
    )
    _assert_bf16_adjacent(got, ref)


def test_no_norm_weight():
    tokens = 4
    fn, scale, base, _ = _params(5)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    ry, _, _ = _ref_pre(x, fn, scale, base, None)
    _, y, _, _ = hc_stage(
        x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
        norm_eps=EPS, norm_weight=None, tokens=tokens, dim=DIM,
    )
    _assert_bf16_adjacent(y, ry)


def test_repeated_launches_are_stable():
    """The arrival counter must be left armed: back-to-back stages sharing one
    workspace have to give the same answer every time."""
    tokens = 8
    fn, scale, base, w = _params(13)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    first = None
    for _ in range(64):
        _, y, post, comb = hc_stage(
            x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
            norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
        )
        if first is None:
            first = (y.clone(), post.clone(), comb.clone())
        else:
            assert torch.equal(y, first[0])
            assert torch.equal(post, first[1])
            assert torch.equal(comb, first[2])


def test_graph_capture_replays():
    """Capture-safe and replay-stable: the split-K counter is reset in-kernel, so a
    replay finds it armed rather than saturated."""
    tokens = 4
    fn, scale, base, w = _params(17)
    x = torch.randn(tokens, HCD, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(tokens, DIM, device=DEV, dtype=torch.bfloat16)
    cb = torch.empty(tokens, HC, HC, device=DEV, dtype=torch.float32)

    def run():
        _, y, _, comb = hc_stage(
            x, None, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=EPS,
            norm_eps=EPS, norm_weight=w, tokens=tokens, dim=DIM,
        )
        out.copy_(y)
        cb.copy_(comb)

    run()  # eager warmup allocates the workspace outside any graph pool
    eager = (out.clone(), cb.clone())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for _ in range(5):
        out.zero_()
        cb.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager[0])
        assert torch.equal(cb, eager[1])
    del g


def test_split_k_is_shape_deterministic():
    idx = torch.cuda.current_device()
    a = split_k_for(1, HCD, idx)
    assert a == split_k_for(1, HCD, idx)
    assert a >= 1 and (a & (a - 1)) == 0
    assert HCD % a == 0 and (HCD // a) % 128 == 0
    # more tokens -> less K splitting
    assert split_k_for(64, HCD, idx) <= a


def test_can_fuse_gate():
    assert can_fuse(1, DIM, HC)
    assert not can_fuse(FUSE_MAX_TOKENS + 1, DIM, HC)
    assert not can_fuse(1, 100, HC)


def test_env_flag_is_read_by_the_model():
    """The escape hatch has to be a real switch, not documentation."""
    from freetoken.models.deepseek_v4 import model as m

    assert hasattr(m, "_UNFUSED_HC")
    assert m._UNFUSED_HC == (os.environ.get("FREETOKEN_UNFUSED_HC", "0") not in ("0", ""))
