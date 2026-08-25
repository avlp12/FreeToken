"""Bit-identity and capture-safety for the fused DSV4 MoE router tail.

The fused kernel (freetoken/kernel/triton/dsv4/router.py) replaces the nine-op
torch chain behind ``Gate.forward``. Every assertion here is ``torch.equal``, not
``allclose``: a 1 ULP drift in the scores can reorder a near-tie in the top-k and
route a token to a different expert, so "close" is not a meaningful guarantee for
a router. ``Gate._reference_route`` is the pre-fusion composition, kept in the
model file precisely so this comparison stays available.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused router is a CUDA Triton kernel"
)

DEV = "cuda"
DIM = 4096
N_EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5
VOCAB = 4096


def _args(hash_layers=0):
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    return DeepseekV4Args(
        dim=DIM,
        n_routed_experts=N_EXPERTS,
        n_activated_experts=TOPK,
        score_func="sqrtsoftplus",
        route_scale=ROUTE_SCALE,
        n_hash_layers=hash_layers,
        vocab_size=VOCAB,
    )


def _gen(seed: int):
    """A private generator, never the global one.

    tests/dsv4/test_hc_fused.py draws some of its inputs from the *ambient* CUDA
    RNG, so any file collected before it that reseeds or consumes the global
    stream silently changes that test's inputs -- and its tolerances are tight
    enough to notice. Nothing here touches the global generator.
    """
    return torch.Generator(device=DEV).manual_seed(seed)


def _gate(hash_layer: bool, seed: int = 0):
    from freetoken.models.deepseek_v4.moe import Gate

    g_rng = _gen(seed)
    args = _args(hash_layers=1 if hash_layer else 0)
    g = Gate(0, args).to(DEV)
    with torch.no_grad():
        g.weight.copy_(
            torch.randn(N_EXPERTS, DIM, device=DEV, dtype=torch.bfloat16, generator=g_rng)
            * 0.02
        )
        if g.bias is not None:
            g.bias.copy_(torch.randn(N_EXPERTS, device=DEV, generator=g_rng) * 0.1)
        if hash_layer:
            g.tid2eid.copy_(
                torch.randint(
                    0, N_EXPERTS, (VOCAB, TOPK), device=DEV, dtype=torch.int64,
                    generator=g_rng,
                )
            )
    return g


def _inputs(m, seed=0, scale=1.0):
    g_rng = _gen(1000 + seed)
    x = torch.randn(m, DIM, device=DEV, dtype=torch.bfloat16, generator=g_rng) * scale
    ids = torch.randint(0, VOCAB, (m,), device=DEV, dtype=torch.int64, generator=g_rng)
    return x, ids


def _both(gate, x, ids, **kw):
    """(fused, reference) for the same inputs, via the public forward + the gate."""
    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32

    fused = gate(x, ids, **kw)
    ref = gate._reference_route(bf16_linear_fp32(x, gate.weight), ids, **kw)
    return fused, ref


# --------------------------------------------------------------------------- #
# bit identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("m", [1, 2, 5, 512])
@pytest.mark.parametrize("scale", [0.01, 1.0, 8.0])
def test_score_route_is_bit_identical(m, scale):
    gate = _gate(hash_layer=False)
    x, ids = _inputs(m, scale=scale)
    (fw, fi), (rw, ri) = _both(gate, x, ids)
    assert torch.equal(fi, ri), f"expert ids diverged (m={m}, scale={scale})"
    assert torch.equal(fw, rw), f"router weights diverged (m={m}, scale={scale})"


@pytest.mark.parametrize("m", [1, 5, 512])
def test_hash_route_is_bit_identical(m):
    gate = _gate(hash_layer=True)
    assert gate.hash and gate.bias is None
    x, ids = _inputs(m)
    (fw, fi), (rw, ri) = _both(gate, x, ids)
    assert torch.equal(fi, ri)
    assert torch.equal(fw, rw)


@pytest.mark.parametrize("m", [1, 5])
def test_return_scores_is_bit_identical(m):
    """The routing tracer's opt-in third output."""
    gate = _gate(hash_layer=False)
    x, ids = _inputs(m)
    (fw, fi, fs), (rw, ri, rs) = _both(gate, x, ids, return_scores=True)
    assert torch.equal(fi, ri)
    assert torch.equal(fw, rw)
    assert torch.equal(fs, rs), "pre-renorm selection scores diverged"


@pytest.mark.parametrize("m", [1, 5])
def test_want_int32_matches_the_cast_it_replaces(m):
    gate = _gate(hash_layer=False)
    x, ids = _inputs(m)
    (fw, fi, f32), (rw, ri, r32) = _both(gate, x, ids, want_int32=True)
    assert f32.dtype == torch.int32 and f32.is_contiguous()
    assert torch.equal(f32, ri.to(torch.int32))
    assert torch.equal(f32, r32)
    # routed_forward rewrites expert ids into slot ids in place, so the int32
    # buffer must not alias anything the caller still needs -- notably the int64
    # ids the prefetcher and the routing tracer read.
    assert f32.data_ptr() != fi.data_ptr()
    before = fi.clone()
    f32.fill_(-1)
    assert torch.equal(fi, before)


def test_selected_scores_are_the_unbiased_ones():
    """The e-score bias steers selection only; weights are built pre-bias.

    A fused router that accidentally renormalises the *biased* scores still
    passes a loose tolerance check on a well-conditioned draw, so pin it: with a
    large bias the two differ by far more than rounding.
    """
    gate = _gate(hash_layer=False)
    with torch.no_grad():
        gate.bias.copy_(torch.arange(N_EXPERTS, device=DEV, dtype=torch.float32))
    x, ids = _inputs(1)
    (fw, fi, fs), (rw, ri, rs) = _both(gate, x, ids, return_scores=True)
    assert torch.equal(fi, ri)
    assert torch.equal(fs, rs)
    assert (fs < gate.bias[fi]).all(), "selection scores look bias-contaminated"


def test_weights_sum_to_route_scale():
    gate = _gate(hash_layer=False)
    x, ids = _inputs(8)
    w, _ = gate(x, ids)
    torch.testing.assert_close(
        w.sum(dim=-1), torch.full((8,), ROUTE_SCALE, device=DEV), rtol=1e-6, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# escape hatch
# --------------------------------------------------------------------------- #


def test_unfused_env_gate_selects_the_reference_path(monkeypatch):
    from freetoken.kernel.triton.dsv4 import router
    from freetoken.models.deepseek_v4 import moe as dsv4_moe

    assert not router.unfused_router()
    assert dsv4_moe._fuse_router("sqrtsoftplus")
    monkeypatch.setenv("FREETOKEN_UNFUSED_ROUTER", "1")
    assert router.unfused_router()
    assert not dsv4_moe._fuse_router("sqrtsoftplus")

    # ... and the gate really produces the reference numbers under it.
    gate = _gate(hash_layer=False)
    x, ids = _inputs(3)
    fw, fi = gate(x, ids)
    monkeypatch.delenv("FREETOKEN_UNFUSED_ROUTER")
    gw, gi = gate(x, ids)
    assert torch.equal(fi, gi)
    assert torch.equal(fw, gw)


def test_unsupported_score_funcs_stay_on_the_torch_chain():
    from freetoken.models.deepseek_v4 import moe as dsv4_moe

    assert not dsv4_moe._fuse_router("softmax")
    assert not dsv4_moe._fuse_router("sigmoid")


# --------------------------------------------------------------------------- #
# capture safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hash_layer", [False, True])
def test_router_is_cuda_graph_capturable(hash_layer):
    """Fixed shapes, no host branch on device data -> capturable, and the replay
    must recompute from the input buffers rather than replaying stale results."""
    gate = _gate(hash_layer=hash_layer)
    x, ids = _inputs(1, seed=3)
    xbuf = x.clone()
    idbuf = ids.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gate(xbuf, idbuf, want_int32=True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gw, gi, g32 = gate(xbuf, idbuf, want_int32=True)

    # Feed inputs the capture never saw; the graph must produce their routing.
    x2, ids2 = _inputs(1, seed=99, scale=3.0)
    xbuf.copy_(x2)
    idbuf.copy_(ids2)
    graph.replay()
    torch.cuda.synchronize()
    rw, ri, r32 = gw.clone(), gi.clone(), g32.clone()

    ew, ei, e32 = gate(x2, ids2, want_int32=True)
    assert torch.equal(ri, ei), "graph replay routed differently from eager"
    assert torch.equal(rw, ew)
    assert torch.equal(r32, e32)


def test_launch_count_collapses():
    """The point of the fusion: one kernel where the chain had nine."""
    from torch.autograd.profiler_util import DeviceType

    gate = _gate(hash_layer=False)
    x, ids = _inputs(1)

    def count(fn):
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn()
            torch.cuda.synchronize()
        return sum(
            e.count
            for e in prof.key_averages()
            if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0
        )

    from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32

    n_fused = count(lambda: gate(x, ids, want_int32=True))
    n_ref = count(
        lambda: gate._reference_route(
            bf16_linear_fp32(x, gate.weight), ids, want_int32=True
        )
    )
    # GEMV + fused tail.
    assert n_fused == 2, f"expected 2 launches, got {n_fused}"
    assert n_ref >= 10, f"reference chain unexpectedly cheap ({n_ref})"
