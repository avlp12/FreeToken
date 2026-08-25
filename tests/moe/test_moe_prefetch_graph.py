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


def _build(
    prefetch: bool,
    topk_limit: int = 0,
    l2_topk: int = 0,
    double_buffer: bool = True,
    late_join: bool = True,
):
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
    # Pinned AFTER registration (which is where the auto limit resolves) and not
    # inherited from the ambient FREETOKEN_PREFETCH_TOPK / _L2_TOPK / _LATE_JOIN: these
    # tests assert on how many rows the speculation pulls and on where the join lands,
    # so every knob has to be a parameter of the test rather than of whatever
    # environment it happens to run in. 0 = no limit / stage off.
    pf.topk_limit = topk_limit
    pf.l2_topk = l2_topk
    pf.double_buffer = double_buffer
    pf.late_join = late_join
    return cache, gates, (pf if prefetch else None)


class _Workload:
    """Everything one decode step touches, so a step is a pure function of `hidden`."""

    def __init__(
        self,
        prefetch: bool,
        mix: float = 0.0,
        mm_iters: int = 1,
        serial: bool = False,
        topk_limit: int = 0,
        l2_topk: int = 0,
        double_buffer: bool = True,
        late_join: bool = True,
    ) -> None:
        self.cache, self.gates, self.pf = _build(
            prefetch, topk_limit, l2_topk, double_buffer, late_join
        )
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
                pf.before_ensure(cache, layer)
            cache.ensure_experts(layer, ids)  # rewrites ids -> slot ids, in place
            cache.copy_missing()
            if pf is not None:
                if self.serial:
                    self._serial_prefetch(layer, h, ids)
                else:
                    pf.schedule(cache, layer, h, self.input_ids, ids)
                pf.before_gemm(cache, layer)
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
        """The same speculative work ``schedule`` does, on the MAIN stream.

        Same gates, same rank limits, same stages, same plan sets, same kernels and the
        same bytes -- only the fork is missing, which is what makes the A/B isolate the
        overlap rather than the extra traffic.
        """
        pf, cache = self.pf, self.cache
        plans = []
        for stage, (dst, limit) in enumerate(pf.stages(cache, layer)):
            gate = self.gates[dst]
            _w, pred = gate(h, self.input_ids)
            if limit:
                pred = pred[..., :limit]
            plan = cache.prefetch_plan_index(layer, stage, pf.double_buffer)
            cache.protect_slots(ids)
            if stage:
                cache.protect_prefetch_plan(plans[-1])
            cache.prefetch_ensure(dst, pred, plan_index=plan)
            plans.append(plan)
        for plan in plans:
            cache.prefetch_copy(plan_index=plan)


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


def _run(prefetch: bool, steps: int, mix: float = 0.0, capture: bool = True, **kw):
    w = _Workload(prefetch, mix=mix, **kw)
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
@pytest.mark.parametrize("l2_topk", [0, 1, 2], ids=["l1-only", "l2-top1", "l2-top2"])
@pytest.mark.parametrize("capture", [True, False], ids=["captured", "eager"])
def test_capture_and_value_identity(mix, l2_topk, capture):
    """A graph with one fork/join per layer captures, replays, and produces exactly
    the prefetch-off values -- with the join down at the expert GEMM.

    ``mix=0`` is the sharpest case: the lookahead predicts exactly, so the rows the
    expert GEMM reads are the ones the FORKED pull wrote rather than ones the main
    stream copied -- it is the configuration that catches a join placed too late.
    ``mix=1`` mostly mispredicts, so it catches wrong admissions corrupting anything.

    Both are run against every L+2 setting: the second stage doubles the plan sets in
    play per layer and adds a second admission that could evict the first's rows, so
    the descriptor lifetimes and the eviction protections are only fully exercised with
    it on. Eager and captured are the same assertion (the fork/join is real machinery in
    both), which also catches a capture-only artifact.
    """
    _woff, off = _run(prefetch=False, steps=6, mix=mix, capture=capture)
    _won, on = _run(prefetch=True, steps=6, mix=mix, capture=capture, l2_topk=l2_topk)
    for i, (a, b) in enumerate(zip(off, on)):
        assert torch.equal(a, b), f"step {i}: prefetch changed the computed values"


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("l2_topk", [0, 1], ids=["l1-only", "l2-top1"])
def test_eager_matches_captured(l2_topk):
    """The captured path and the eager path agree (no capture-only artifact)."""
    _w1, eager = _run(prefetch=True, steps=4, mix=1.0, capture=False, l2_topk=l2_topk)
    _w2, captured = _run(prefetch=True, steps=4, mix=1.0, capture=True, l2_topk=l2_topk)
    for i, (a, b) in enumerate(zip(eager, captured)):
        assert torch.equal(a, b), f"step {i}: eager and captured prefetch disagree"


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("mix", [0.0, 1.0], ids=["exact-lookahead", "mispredicting"])
def test_early_join_still_correct(mix):
    """FREETOKEN_PREFETCH_LATE_JOIN=0 (the old placement) must remain value-identical.

    The knob exists so the placement can be A/B'd at the server; both placements have
    to be correct, or the A/B measures a bug instead of a window.
    """
    _woff, off = _run(prefetch=False, steps=6, mix=mix)
    _won, on = _run(prefetch=True, steps=6, mix=mix, l2_topk=1, late_join=False)
    for i, (a, b) in enumerate(zip(off, on)):
        assert torch.equal(a, b), f"step {i}: early-join prefetch changed the values"


@CUDA
def test_plan_sets_alternate_by_source_layer_parity():
    """The descriptor lifetime argument, checked directly rather than by racing.

    A plan set may only be rewritten once every pull that reads it has retired. Layer
    L's join covers layer L-1's fork, so the first layer allowed to reuse layer L's
    descriptors is L+2 -- which is exactly what indexing by ``L % 2`` gives, and exactly
    what single buffering does NOT: there, L+1 already collides with L.
    """
    from freetoken.moe.offload_cache import PREFETCH_PLAN_SETS, OffloadMoeCache

    idx = OffloadMoeCache.prefetch_plan_index
    for layer in range(8):
        for stage in range(2):
            assert idx(layer, stage) != idx(layer + 1, stage), (
                f"layer {layer} and {layer + 1} share a plan set: the pull forked at "
                f"{layer} is still in flight when {layer + 1} stages its own"
            )
            assert idx(layer, stage) == idx(layer + 2, stage)
        # The two stages of one layer are separate plans too: both pulls are in flight
        # together on the branch.
        assert idx(layer, 0) != idx(layer, 1)
    # Single buffering is the racy layout the double buffer replaced.
    assert idx(0, 0, False) == idx(1, 0, False)
    assert len({idx(L, s) for L in range(4) for s in range(2)}) == PREFETCH_PLAN_SETS


class _PlanLifetimeAudit:
    """Host-side ledger of when each descriptor set is written vs. when its pull retires.

    A prefetch plan is LIVE from the moment a speculative ensure writes it (the pull that
    reads it is forked immediately after, in the same ``schedule``) until the join that
    orders the main stream behind that pull. Writing a live plan is precisely the
    unsound condition a late join has without parity indexing -- the pull is left reading
    descriptors that now describe a different layer's rows.

    Auditing it here rather than watching for corrupted values is deliberate. The write
    and the pull's read are a genuine data race, so whether it changes any output depends
    on how quickly the branch kernel gets scheduled: on an idle GPU the pull's blocks
    load their indices microseconds after launch and win, which is why this toy never
    corrupts, and on a loaded one they do not, which is why the server did (4 of 4 runs).
    A race that only fires under contention is not something to assert on -- but the
    ordering that makes it possible is exact, host-side, and evaluated at capture time.
    """

    def __init__(self, workload) -> None:
        self.live: dict[int, int] = {}  # plan set -> the layer whose pull is reading it
        self.violations: list[tuple[int, int, int]] = []
        cache, pf = workload.cache, workload.pf
        inner_ensure, inner_join = cache.prefetch_ensure, pf.join

        def ensure(layer_id, ids, plan_index=0):
            if plan_index in self.live:
                self.violations.append((plan_index, self.live[plan_index], layer_id))
            inner_ensure(layer_id, ids, plan_index=plan_index)
            self.live[plan_index] = layer_id

        def join(c, layer_id):
            armed = pf._armed.get(layer_id)
            inner_join(c, layer_id)
            if armed is not None:
                src, n_stages = armed
                for stage in range(n_stages):
                    self.live.pop(
                        cache.prefetch_plan_index(src, stage, pf.double_buffer), None
                    )

        cache.prefetch_ensure = ensure
        pf.join = join


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("l2_topk", [0, 1], ids=["l1-only", "l2-top1"])
def test_double_buffering_is_what_makes_the_late_join_sound(l2_topk):
    """No descriptor set is rewritten while a pull reading it is still unjoined --
    and collapsing the parity index breaks exactly that, which is the whole reason
    the join could not be moved before."""
    w = _Workload(True, mix=0.0, l2_topk=l2_topk, double_buffer=True)
    audit = _PlanLifetimeAudit(w)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            w.step()
    torch.cuda.synchronize()
    assert audit.violations == [], (
        "a live prefetch plan was overwritten: " + str(audit.violations)
    )

    single = _Workload(True, mix=0.0, l2_topk=l2_topk, double_buffer=False)
    single_audit = _PlanLifetimeAudit(single)
    with torch.cuda.stream(stream):
        for _ in range(3):
            single.step()
    torch.cuda.synchronize()
    print(
        f"\nsingle-buffered late join: {len(single_audit.violations)} live-plan "
        f"overwrites (plan, reader layer, clobbering layer): "
        f"{single_audit.violations[:4]}"
    )
    assert single_audit.violations, (
        "the audit found nothing to complain about in the single-buffered layout, so it "
        "would not have caught the bug the double buffering fixes"
    )
    # ... and every one of them is the next layer clobbering its predecessor's plan.
    assert all(clobber == reader + 1 for _p, reader, clobber in single_audit.violations)


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
def test_rank_limit_trades_misses_for_bytes():
    """FREETOKEN_PREFETCH_TOPK must actually reduce the rows the speculation pulls.

    On a bandwidth-saturated link that trade is the whole ballgame -- the server-level
    finding was that speculating on all six predictions moves 37.5% more bytes and
    cancels the bandwidth the fork wins back. ``pf_pulled_per_layer`` (real misses
    removed) and ``missing_per_layer`` (misses left) are the two halves of it, and
    both come from the same stats path the server dumps.
    """
    from flashlib.kernels.slot_cache import Stat

    def measure(topk_limit: int) -> tuple[float, float]:
        w = _Workload(True, mix=0.5, topk_limit=topk_limit)
        w.cache.collect_stats = True
        g = torch.Generator(device="cpu").manual_seed(11)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                w.step()
            w.cache.reset()
            w.cache.lru_stats.zero_()
            w.cache.prefetch_stats.zero_()
            for _ in range(8):
                w.hidden.copy_(torch.randn(1, DIM, generator=g).to(w.dev).bfloat16())
                w.step()
        torch.cuda.synchronize()
        stats = w.cache.decode_miss_stats()
        return stats["missing_per_layer"], stats["pf_pulled_per_layer"]

    full_miss, full_pull = measure(0)
    lim_miss, lim_pull = measure(2)
    print(
        f"\nrank limit: all-{TOP_K} -> misses {full_miss:.2f} + pulled {full_pull:.2f} "
        f"= {full_miss + full_pull:.2f} rows; "
        f"top-2 -> {lim_miss:.2f} + {lim_pull:.2f} = {lim_miss + lim_pull:.2f} rows"
    )
    assert lim_pull < full_pull, "the rank limit did not reduce speculative pulls"
    assert lim_miss > full_miss, "fewer predictions must leave more real misses"
    assert Stat.MISS == 1  # the column pf_pulled_per_layer reads


@CUDA
@pytest.mark.slow
def test_l2_stage_pulls_a_second_layer_ahead():
    """The L+2 chain is wired: it predicts two layers out and moves those rows.

    What this does NOT assert is that the stage removes real misses, because in this toy
    it does not, and the reason is a property of the toy rather than of the feature. An
    admission made at layer L for layer L+2 has to survive four intervening
    ``lru_ensure`` calls; with 40 slots for 64 ids each of those calls is admitting ~6
    rows, so the two-layer-early row is the oldest thing in the cache by the time it is
    wanted and LRU takes it back before it is ever read. Production has 740 slots against
    ~9 rows per call -- an order of magnitude more residency across the same window --
    which is a different regime, not a smaller version of this one.

    Whether the stage pays for its bytes is therefore a question for
    ``benchmarks/moe_prefetch_overlap.py``, which is shaped like the real model. Here we
    pin the mechanism (and value identity is covered above); the miss delta is printed
    rather than asserted so a regression that silently stops predicting still shows.
    """

    def measure(l2_topk: int) -> dict:
        w = _Workload(True, mix=0.5, topk_limit=1, l2_topk=l2_topk)
        w.cache.collect_stats = True
        g = torch.Generator(device="cpu").manual_seed(11)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                w.step()
            w.cache.reset()
            w.cache.lru_stats.zero_()
            w.cache.prefetch_stats.zero_()
            for _ in range(8):
                w.hidden.copy_(torch.randn(1, DIM, generator=g).to(w.dev).bfloat16())
                w.step()
        torch.cuda.synchronize()
        return w.cache.decode_miss_stats()

    one, two = measure(0), measure(1)
    print(
        "\nL+2 chain (4-layer toy, 40 slots):"
        f"\n  L+1 only   stages/layer {one['pf_stages_per_layer']:.2f}  "
        f"predicted {one['pf_predicted_per_layer']:.2f}  pulled {one['pf_pulled_per_layer']:.2f}"
        f"  real misses {one['missing_per_layer']:.2f}"
        f"\n  +L+2 top-1 stages/layer {two['pf_stages_per_layer']:.2f}  "
        f"predicted {two['pf_predicted_per_layer']:.2f}  pulled {two['pf_pulled_per_layer']:.2f}"
        f"  real misses {two['missing_per_layer']:.2f}"
    )
    # Layers 0 and 1 are the only ones with an L+2 target in a 4-layer stack, so the
    # extra stage runs on half of them.
    assert two["pf_stages_per_layer"] > one["pf_stages_per_layer"], "no second stage ran"
    assert two["pf_predicted_per_layer"] > one["pf_predicted_per_layer"], (
        "the L+2 Gate predicted nothing"
    )
    assert two["pf_pulled_per_layer"] > one["pf_pulled_per_layer"], (
        "the L+2 stage pulled no extra rows (is the second plan reaching the branch?)"
    )


@CUDA
@pytest.mark.slow
def test_l2_stage_off_is_the_l1_graph():
    """FREETOKEN_PREFETCH_L2_TOPK=0 must disable the stage cleanly: no second Gate, no
    second admission, no second pull -- the L+1-only behaviour, exactly."""
    w = _Workload(True, mix=0.5, l2_topk=0)
    assert all(len(w.pf.stages(w.cache, L)) <= 1 for L in range(NUM_LAYERS))
    w.cache.collect_stats = True
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        w.step()
    torch.cuda.synchronize()
    # One speculative ensure per prefetching layer, never two.
    calls = int(w.cache.prefetch_stats[:, 2].sum())
    assert calls == sum(w.pf.prefetches(w.cache, L) for L in range(NUM_LAYERS)), calls
    # Only parity-0/1 STAGE-0 plan sets were ever staged.
    staged = [i for i, layer in enumerate(w.cache._prefetch_staged) if layer is not None]
    assert all(i % 2 == 0 for i in staged), staged


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
