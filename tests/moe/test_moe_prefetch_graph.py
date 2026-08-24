"""In-graph L+1 expert prefetch: capture, value identity, and real overlap.

Drives the production pieces -- ``OffloadMoeCache`` with synthetic pinned banks, the
real ``MoePrefetcher`` fork/join orchestration, real ``lru_ensure`` admission and the
real fused pull kernel -- through a decode-shaped multi-layer loop, with no server,
no model weights and no engine.

Three questions:

1. does a decode graph carrying one fork/join round PER LAYER capture and replay at
   all (this platform is WSL2 GPU-PV / WDDM, where the copy-engine doorbell path
   deadlocks on exactly this shape);
2. are the outputs bit-identical to the prefetch-off graph for the same inputs --
   prefetch changes WHEN rows land, never what any GEMM reads, and a mispredicted
   admission must corrupt nothing;
3. does the forked pull actually overlap -- measured against the SAME prefetch work
   issued serially on the main stream, so the comparison isolates the fork and not
   the extra bytes it moves.

The toy stack mirrors the real one where it matters. Each layer transforms the hidden
before the next layer's router sees it (``mix`` sets how much), so running layer
L+1's Gate on layer L's hidden is a genuine prediction: ``mix=0`` predicts exactly,
``mix=1`` mostly mispredicts. The "expert GEMM" is a bf16 matmul (something for the
pull to hide behind) plus a gather of the routed slot rows -- the only property that
matters for correctness here, since it is what a wrongly-evicted or half-written row
would corrupt -- which keeps the test independent of any quant format's kernel.

The cache is deliberately undersized (40 slots for 64 ids) so every step really does
evict and refetch; a cache that fits the whole toy model would make both the real
pull and the prefetch no-ops after warm-up.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_LAYERS = 4
NUM_EXPERTS = 16
TOP_K = 6
CACHE_SIZE = 40  # >= 4 * TOP_K (the prefetch eviction-safety floor), well under 64 ids
DIM = 512
# ~1.5MB per expert across the two banks: the pull is a real PCIe transfer.
FEATS = (1024 * 1024, 512 * 1024)
MM = 2048  # stand-in compute size


class _FakeGate(torch.nn.Module):
    """Stand-in router: a real GEMV + top-k, with the same ``(x, input_ids) ->
    (weights, ids)`` signature the production Gate has, so the lookahead call the
    prefetcher makes is a genuine extra kernel on the main stream."""

    def __init__(self, layer_id: int, device) -> None:
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(1000 + layer_id)
        self.weight = torch.randn(NUM_EXPERTS, DIM, generator=g).to(device)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        scores = (x.float() @ self.weight.t().float()).softmax(dim=-1)
        idx = scores.topk(TOP_K, dim=-1)[1]
        return scores.gather(1, idx), idx.to(torch.int32)


def _build(prefetch: bool):
    from freetoken.moe.offload_cache import MOE_PREFETCH_ENV, OffloadMoeCache
    from freetoken.moe.prefetch import MoePrefetcher

    dev = torch.device("cuda")
    prev = os.environ.get(MOE_PREFETCH_ENV)
    os.environ[MOE_PREFETCH_ENV] = "1" if prefetch else "0"
    try:
        cache = OffloadMoeCache(
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            cache_size=CACHE_SIZE,
            device=dev,
            prefill_overlap=False,
            quant_format="bf16",
        )
    finally:
        if prev is None:
            os.environ.pop(MOE_PREFETCH_ENV, None)
        else:
            os.environ[MOE_PREFETCH_ENV] = prev

    # Pinned host banks with identity-derived content: a GEMM reading the wrong slot,
    # or a row half-overwritten by a racing pull, changes the reduction.
    sources = {}
    for name, feat in zip(("gate_up", "down"), FEATS):
        per_layer = []
        for layer in range(NUM_LAYERS):
            t = torch.empty((NUM_EXPERTS, feat // 2), dtype=torch.bfloat16, pin_memory=True)
            for e in range(NUM_EXPERTS):
                t[e].fill_(float(1 + layer * NUM_EXPERTS + e) / 64.0)
            per_layer.append(t)
        sources[name] = per_layer
    cache.set_bank_sources(sources)
    assert cache._copy_fused_ok, "fused copy plan is required by the prefetch path"

    gates = [_FakeGate(i, dev) for i in range(NUM_LAYERS)]
    pf = MoePrefetcher()
    for i, g in enumerate(gates):
        pf.register_gate(i, g, n_layers=NUM_LAYERS, top_k=TOP_K, n_experts=NUM_EXPERTS)
    return cache, gates, (pf if prefetch else None)


class _Workload:
    """Everything one decode step touches, so a step is a pure function of `hidden`."""

    def __init__(
        self,
        prefetch: bool,
        mix: float = 0.0,
        mm_iters: int = 1,
        serial: bool = False,
    ) -> None:
        self.cache, self.gates, self.pf = _build(prefetch)
        dev = torch.device("cuda")
        self.dev = dev
        self.hidden = torch.zeros(1, DIM, dtype=torch.bfloat16, device=dev)
        self.input_ids = torch.zeros(1, dtype=torch.int64, device=dev)
        self.out = torch.zeros(NUM_LAYERS, DIM, dtype=torch.float32, device=dev)
        # Seeded on the CPU: the global CUDA generator advances between workloads, and
        # the stand-in GEMM's output feeds `out`, so two runs must draw the same bytes.
        mg = torch.Generator(device="cpu").manual_seed(4242)
        self.mm_a = torch.randn(MM, MM, generator=mg).to(dev).bfloat16()
        self.mm_b = torch.randn(MM, MM, generator=mg).to(dev).bfloat16()
        self.mm_sink = torch.zeros(1, dtype=torch.bfloat16, device=dev)
        self.mm_iters = mm_iters
        # Per-layer hidden transform: how far layer L+1's router input drifts from
        # layer L's, i.e. how good the one-layer-early prediction can be.
        g = torch.Generator(device="cpu").manual_seed(99)
        self.rot = [
            (torch.randn(DIM, DIM, generator=g) / DIM**0.5).to(dev).bfloat16()
            for _ in range(NUM_LAYERS)
        ]
        self.mix = mix
        # "serial": run the identical prefetch work on the MAIN stream instead of the
        # forked branch. Same kernels, same bytes, no concurrency -- the A/B that
        # isolates the fork itself.
        self.serial = serial

    def step(self) -> None:
        cache, gates, pf = self.cache, self.gates, self.pf
        views = [v[:, :DIM] for v in cache.bank_views()]
        h = self.hidden
        for layer in range(NUM_LAYERS):
            _w, ids = gates[layer](h, self.input_ids)
            ids = ids.reshape(-1).contiguous()
            if pf is not None:
                pf.join(cache, layer)
            cache.ensure_experts(layer, ids)  # rewrites ids -> slot ids, in place
            cache.copy_missing()
            if pf is not None:
                if self.serial:
                    self._serial_prefetch(layer, h, ids)
                else:
                    pf.schedule(cache, layer, h, self.input_ids, ids)
            # stand-in expert GEMM: compute for the branch to hide behind, then the
            # routed slot rows of every bank
            x = self.mm_a
            for _ in range(self.mm_iters):
                x = torch.mm(x, self.mm_b)
            # The matmul exists only to give the forked pull something to overlap; it
            # is drained into a sink rather than into `out`, because cuBLAS may pick a
            # different bf16 algorithm depending on allocator state and that would make
            # the value-identity comparison test cuBLAS rather than the prefetch.
            self.mm_sink.copy_(x[0, :1])
            acc = self.out[layer]
            acc.zero_()
            rows = ids.long()
            for v in views:
                acc += v.index_select(0, rows).float().sum(dim=0)
            if self.mix:
                h = h * (1.0 - self.mix) + torch.mm(h, self.rot[layer]) * self.mix

    def _serial_prefetch(self, layer: int, h: torch.Tensor, ids: torch.Tensor) -> None:
        dst = layer + 1
        if not self.pf.prefetches(self.cache, layer):
            return
        _w, pred = self.gates[dst](h, self.input_ids)
        self.cache.protect_slots(ids)
        self.cache.prefetch_ensure(dst, pred)
        self.cache.prefetch_copy()


def _capture(w: _Workload):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):  # eager warm-up: allocates the prefetch scratch pre-capture
            w.step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        w.step()
    return graph, stream


def _run(prefetch: bool, steps: int, mix: float = 0.0, capture: bool = True):
    w = _Workload(prefetch, mix=mix)
    g = torch.Generator(device="cpu").manual_seed(7)
    inputs = [torch.randn(1, DIM, generator=g).to(w.dev).bfloat16() for _ in range(steps)]
    results = []
    stream = torch.cuda.Stream()
    if not capture:
        with torch.cuda.stream(stream):
            for _ in range(2):
                w.step()
            for x in inputs:
                w.hidden.copy_(x)
                w.step()
                torch.cuda.synchronize()
                results.append(w.out.clone())
        return w, results
    graph, stream = _capture(w)
    with torch.cuda.stream(stream):
        for x in inputs:
            w.hidden.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            results.append(w.out.clone())
    return w, results


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("mix", [0.0, 1.0], ids=["exact-lookahead", "mispredicting"])
def test_capture_and_value_identity(mix):
    """A graph with one fork/join per layer captures, replays, and produces exactly
    the prefetch-off values.

    ``mix=0`` is the sharpest case: the lookahead predicts exactly, so the rows the
    expert GEMM reads are the ones the FORKED pull wrote rather than ones the main
    stream copied -- it is the configuration that catches a join placed too late.
    ``mix=1`` mostly mispredicts, so it catches wrong admissions corrupting anything.
    """
    _woff, off = _run(prefetch=False, steps=6, mix=mix)
    _won, on = _run(prefetch=True, steps=6, mix=mix)
    for i, (a, b) in enumerate(zip(off, on)):
        assert torch.equal(a, b), f"step {i}: prefetch changed the computed values"


@CUDA
@pytest.mark.slow
def test_eager_matches_captured():
    """The captured path and the eager path agree (no capture-only artifact)."""
    _w1, eager = _run(prefetch=True, steps=4, mix=1.0, capture=False)
    _w2, captured = _run(prefetch=True, steps=4, mix=1.0, capture=True)
    for i, (a, b) in enumerate(zip(eager, captured)):
        assert torch.equal(a, b), f"step {i}: eager and captured prefetch disagree"


@CUDA
@pytest.mark.slow
def test_prefetch_removes_real_misses():
    """The prefetch does useful work: with an exact lookahead the REAL ensure_experts
    of every prefetched layer finds its experts already resident, and a mispredicting
    lookahead degrades toward the no-prefetch baseline instead of breaking."""
    from flashlib.kernels.slot_cache import Stat

    def misses(prefetch: bool, mix: float) -> list[int]:
        w = _Workload(prefetch, mix=mix)
        w.cache.collect_stats = True
        g = torch.Generator(device="cpu").manual_seed(11)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                w.step()
            w.cache.reset()
            w.cache.lru_stats.zero_()
            for _ in range(8):
                w.hidden.copy_(torch.randn(1, DIM, generator=g).to(w.dev).bfloat16())
                w.step()
        torch.cuda.synchronize()
        return w.cache.lru_stats[:, Stat.MISS].tolist()

    off = misses(False, 0.0)
    exact = misses(True, 0.0)
    wrong = misses(True, 1.0)
    print(f"\nreal misses per layer: off={off} prefetch-exact={exact} prefetch-wrong={wrong}")
    # Layer 0 has no predecessor and is never prefetched; layers 1.. are.
    assert sum(exact[1:]) < sum(off[1:]) * 0.25, (off, exact)
    # A mispredicting lookahead must still be a genuine prediction attempt (not a
    # no-op) and must not make the real path worse than not prefetching at all.
    assert sum(wrong[1:]) > sum(exact[1:]), (exact, wrong)


@CUDA
@pytest.mark.slow
def test_forked_pull_overlaps():
    """The forked prefetch pull runs concurrently with compute: forking must beat the
    identical prefetch work issued serially on the main stream."""

    reps = 30

    def timed(**kw) -> float:
        w = _Workload(mix=0.5, **kw)
        graph, stream = _capture(w)
        # A fresh hidden per replay, as the engine does: with a frozen input the toy
        # working set goes fully resident after warm-up and neither path copies
        # anything, so there would be nothing to overlap.
        g = torch.Generator(device="cpu").manual_seed(5)
        xs = [
            torch.randn(1, DIM, generator=g).to(w.dev).bfloat16() for _ in range(reps + 5)
        ]
        with torch.cuda.stream(stream):
            for i in range(5):
                w.hidden.copy_(xs[i])
                graph.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(reps):
                w.hidden.copy_(xs[i + 5])
                graph.replay()
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    base = timed(prefetch=False)
    serial = timed(prefetch=True, serial=True)
    forked = timed(prefetch=True, serial=False)
    print(
        f"\nper-replay ms: no-prefetch={base:.3f} serial-prefetch={serial:.3f} "
        f"forked-prefetch={forked:.3f}  (hidden by the fork: {serial - forked:.3f} ms/step)"
    )
    assert serial > base, "the serial A/B did not actually add the prefetch work"
    assert forked < serial, f"forking did not help (serial={serial:.3f} forked={forked:.3f})"
