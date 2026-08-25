"""Where the MoE prefetch join should go, measured on a decode loop shaped like the
real model: 43 layers, synthetic pinned banks, one CUDA graph per configuration.

The question this exists to answer is narrow. In-graph L+1 prefetch ships and is worth
+13.5%, but CUPTI says the copy stream is only ~26% hidden, and there are exactly two
structural reasons for that:

* the branch join sits before ``ensure_experts(L+1)`` rather than before ``GEMM(L+1)``,
  which throws away the overlap against that ensure and against ``copy_missing(L+1)``;
* the lookahead only ever predicts one layer out, so a pull gets ~1.1ms of window
  even though the same hidden predicts two layers out at 65.3% recall.

Both are now implemented (parity-indexed descriptors make the late join sound; the L+2
stage is behind ``FREETOKEN_PREFETCH_L2_TOPK``). This harness measures what they are
worth in ms/step BEFORE anything is claimed at the server, because both of them spend
bytes on a link that is already the bottleneck and neither is obviously a win.

What it is NOT: a tok/s prediction. The synthetic expert rows are far smaller than the
model's 9.19 MiB and the stand-in GEMM is a matmul, so the absolute numbers mean
nothing. The pull:compute ratio per layer is what is tuned to resemble production, and
it is printed, so the deltas can be read as "does moving the join buy overlap here".

Run::

    python benchmarks/moe_prefetch_overlap.py            # the headline table
    python benchmarks/moe_prefetch_overlap.py --rows-only --skew 0.03   # retune the
                                                         # routing to a target miss rate
    python benchmarks/moe_prefetch_overlap.py --topk 6 --mm 1024        # other regimes

Costs ~5.4 GiB of pinned host memory and 0.72 GiB of GPU at the defaults.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

# Shaped like DSV4-Flash: 43 MoE layers, top-6 routing. The expert rows are deliberately
# ~9x smaller than the model's 9.19 MiB so the slot cache fits well under 1.5 GB of GPU
# (a server may be live on this box); --mm is then sized to put the per-layer
# pull:compute ratio in production's neighbourhood rather than the row size.
#
# The defaults are not arbitrary -- they are the point at which this harness reproduces
# the server's regime on the one measurement that proves it: forking the pull instead of
# issuing it on the main stream is worth +12.3% here against the model's +13.5%, and the
# extra speculative bytes cancel it back to ~0 exactly as the production A/B did. A
# harness that does not reproduce THAT cannot be trusted to grade a change to the
# overlap, which is why "pull on main (A/B)" is a configuration and not a footnote.
NUM_LAYERS = 43
TOP_K = 6
DIM = 1024
# Bytes per expert row, split across the two banks 3:1. Set from --row-kib; the default
# is chosen with --experts/--slots so that the realized miss count per layer lands near
# the model's measured 2.74, which is the number that decides how much there is to hide.
FEATS = (96 * 1024, 32 * 1024)


class _Gate(torch.nn.Module):
    """Stand-in router with production-like skew: a GEMV plus a fixed per-layer expert
    bias, so a few experts are hot and the LRU has something to actually cache. A
    uniformly random router would miss on nearly every route and make every
    configuration look identical."""

    def __init__(self, layer_id: int, n_experts: int, device, skew: float) -> None:
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(1000 + layer_id)
        w = torch.randn(n_experts, DIM, generator=g)
        # Both sides normalized so the hidden-driven term is a cosine similarity in
        # [-1, 1] and ``skew`` is a real dial against it: 0 routes purely on the hidden
        # (maximum churn, nothing caches), large values pin the same hot experts every
        # step (nothing ever misses). Neither extreme resembles the model, and getting
        # this wrong is how a harness ends up measuring an empty cache or a full one.
        self.weight = (w / w.norm(dim=1, keepdim=True)).to(device)
        pop = torch.randn(n_experts, generator=g).sort(descending=True)[0]
        self.bias = (pop * skew).to(device)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        h = x.float()
        scores = (h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6)) @ self.weight.t()
        scores = scores + self.bias
        idx = scores.topk(TOP_K, dim=-1)[1]
        return scores.gather(1, idx), idx.to(torch.int32)


class Harness:
    """One cache, one set of banks, one prefetcher -- and a graph per configuration.

    Everything is shared on purpose: four independent workloads would need four times
    the pinned host memory and four slot caches, and the timing comparison only needs
    the traced code to differ.
    """

    def __init__(self, args) -> None:
        from freetoken.moe.offload_cache import MOE_PREFETCH_ENV, OffloadMoeCache
        from freetoken.moe.prefetch import MoePrefetcher

        self.args = args
        dev = torch.device("cuda")
        self.dev = dev
        # A production server may be live on this box. Refuse to size the slot cache
        # into its headroom rather than discovering it as an OOM in the server's log.
        row_bytes = sum(FEATS)
        want = args.slots * row_bytes + 4 * args.mm * args.mm  # cache + the stand-in GEMM
        budget = args.max_gpu_gib * 2**30
        assert want < budget, (
            f"this configuration wants {want / 2**30:.2f} GiB of GPU for the slot cache "
            f"alone, over the {args.max_gpu_gib} GiB budget; lower --slots"
        )
        free = torch.cuda.mem_get_info()[0]
        assert want < free * 0.6, (
            f"only {free / 2**30:.2f} GiB free on the device; refusing to take "
            f"{want / 2**30:.2f} GiB of it"
        )
        prev = os.environ.get(MOE_PREFETCH_ENV)
        os.environ[MOE_PREFETCH_ENV] = "1"
        try:
            self.cache = OffloadMoeCache(
                num_layers=NUM_LAYERS,
                num_experts=args.experts,
                cache_size=args.slots,
                device=dev,
                prefill_overlap=False,
                quant_format="bf16",
            )
        finally:
            if prev is None:
                os.environ.pop(MOE_PREFETCH_ENV, None)
            else:
                os.environ[MOE_PREFETCH_ENV] = prev

        sources = {}
        for name, feat in zip(("gate_up", "down"), FEATS):
            per_layer = []
            for layer in range(NUM_LAYERS):
                t = torch.empty(
                    (args.experts, feat // 2), dtype=torch.bfloat16, pin_memory=True
                )
                t.fill_(float(layer + 1) / 64.0)
                per_layer.append(t)
            sources[name] = per_layer
        self.cache.set_bank_sources(sources)
        assert self.cache._copy_fused_ok, "the fused multi-bank copy plan is required"
        self.cache.collect_stats = True

        self.gates = [_Gate(i, args.experts, dev, args.skew) for i in range(NUM_LAYERS)]
        self.pf = MoePrefetcher()
        for i, g in enumerate(self.gates):
            self.pf.register_gate(
                i, g, n_layers=NUM_LAYERS, top_k=TOP_K, n_experts=args.experts
            )
        self.pf.topk_limit = args.topk

        self.hidden = torch.zeros(1, DIM, dtype=torch.bfloat16, device=dev)
        self.input_ids = torch.zeros(1, dtype=torch.int64, device=dev)
        self.out = torch.zeros(NUM_LAYERS, DIM, dtype=torch.float32, device=dev)
        mg = torch.Generator(device="cpu").manual_seed(4242)
        mm = args.mm
        self.mm_a = torch.randn(mm, mm, generator=mg).to(dev).bfloat16()
        self.mm_b = torch.randn(mm, mm, generator=mg).to(dev).bfloat16()
        self.mm_sink = torch.zeros(1, dtype=torch.bfloat16, device=dev)
        g = torch.Generator(device="cpu").manual_seed(99)
        self.rot = [
            (torch.randn(DIM, DIM, generator=g) / DIM**0.5).to(dev).bfloat16()
            for _ in range(NUM_LAYERS)
        ]
        # 0 predicts L+1 exactly, 1 drifts the hidden fully between layers. The default
        # sits where the one-layer-early prediction is good but not free, which is the
        # regime the routing tracer measured on the real model (71.5% recall at L+1).
        self.mix = args.mix
        self.prefetch_on = True
        self.serial = False

    def step(self) -> None:
        cache, pf = self.cache, self.pf
        views = [v[:, :DIM] for v in cache.bank_views()]
        h = self.hidden
        for layer in range(NUM_LAYERS):
            _w, ids = self.gates[layer](h, self.input_ids)
            ids = ids.reshape(-1).contiguous()
            if self.prefetch_on:
                pf.before_ensure(cache, layer)
            cache.ensure_experts(layer, ids)
            cache.copy_missing()
            if self.prefetch_on:
                if self.serial:
                    self._serial_prefetch(layer, h, ids)
                else:
                    pf.schedule(cache, layer, h, self.input_ids, ids)
                pf.before_gemm(cache, layer)
            x = torch.mm(self.mm_a, self.mm_b)
            self.mm_sink.copy_(x[0, :1])
            acc = self.out[layer]
            acc.zero_()
            rows = ids.long()
            for v in views:
                acc += v.index_select(0, rows).float().sum(dim=0)
            if self.mix:
                h = h * (1.0 - self.mix) + torch.mm(h, self.rot[layer]) * self.mix

    def _serial_prefetch(self, layer: int, h: torch.Tensor, ids: torch.Tensor) -> None:
        """The identical speculative work, issued on the MAIN stream instead of forked.

        Same gates, same admissions, same kernels, same bytes over the same link -- only
        the concurrency is gone. Comparing against this rather than against prefetch-off
        is what separates "the fork overlaps" from "the extra rows happened to help".
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

    def configure(
        self, prefetch: bool, late_join: bool, l2_topk: int, serial: bool = False
    ) -> None:
        self.prefetch_on = prefetch
        self.serial = serial
        self.pf.late_join = late_join
        self.pf.l2_topk = l2_topk

    def capture(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):  # eager warm-up: allocates the prefetch scratch
                self.step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.step()
        return graph, stream


def _inputs(dev, n, seed=5):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(1, DIM, generator=g).to(dev).bfloat16() for _ in range(n)]


def _rows(h: Harness) -> dict:
    """Realized per-layer PCIe row counts over one measurement window."""
    h.cache.reset_stats()
    stream = torch.cuda.Stream()
    xs = _inputs(h.dev, 24, seed=17)
    with torch.cuda.stream(stream):
        for x in xs:
            h.hidden.copy_(x)
            h.step()
    torch.cuda.synchronize()
    s = h.cache.decode_miss_stats()
    return {
        "real": s["missing_per_layer"],
        "spec": s.get("pf_pulled_per_layer", 0.0),
        "total": s.get("pf_rows_per_layer_total", s["missing_per_layer"]),
        "stages": s.get("pf_stages_per_layer", 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--slots", type=int, default=740, help="GPU slot cache size")
    ap.add_argument("--row-kib", type=int, default=1024, help="KiB per expert row")
    ap.add_argument("--mm", type=int, default=2048, help="stand-in expert GEMM size")
    ap.add_argument("--mix", type=float, default=0.35, help="hidden drift per layer")
    # 0.04 was picked by sweeping --rows-only until the realized row counts matched the
    # model's: 2.573 real rows/layer with prefetch off (model: 2.742) and 1.685 + 0.998
    # at L+1 top-3 (model: 1.689 + 1.226). The real-miss column lands within 0.3%, which
    # is what makes the ms/step deltas below worth reading at all.
    ap.add_argument("--skew", type=float, default=0.04, help="router popularity skew")
    ap.add_argument("--topk", type=int, default=3, help="L+1 speculation rank limit")
    ap.add_argument("--reps", type=int, default=40, help="graph replays per timing")
    ap.add_argument("--rounds", type=int, default=9, help="interleaved timing rounds")
    ap.add_argument("--max-gpu-gib", type=float, default=1.5, help="slot-cache budget")
    ap.add_argument(
        "--rows-only", action="store_true", help="report row counts, skip the timing"
    )
    args = ap.parse_args()

    global FEATS
    FEATS = (args.row_kib * 768, args.row_kib * 256)  # 3:1 across the two banks

    torch.cuda.init()
    h = Harness(args)

    # (label, prefetch, late_join, l2_topk, serial)
    configs = [
        ("prefetch off", False, False, 0, False),
        ("pull on main (A/B)", True, False, 0, True),
        ("early join (shipped)", True, False, 0, False),
        ("late join", True, True, 0, False),
        ("late join + L+2 top-1", True, True, 1, False),
        ("late join + L+2 top-2", True, True, 2, False),
    ]

    print(
        f"\n{NUM_LAYERS} layers x {args.experts} experts, top-{TOP_K}, "
        f"{args.slots} slots, L+1 speculation top-{args.topk}, mix={args.mix}"
    )
    row_mib = sum(FEATS) / 1024 / 1024
    print(
        f"expert row {row_mib:.2f} MiB across 2 banks; "
        f"slot cache {args.slots * row_mib / 1024:.2f} GiB, "
        f"host banks {NUM_LAYERS * args.experts * row_mib / 1024:.2f} GiB pinned"
    )

    graphs, rows = {}, {}
    for label, prefetch, late, l2, serial in configs:
        h.configure(prefetch, late, l2, serial)
        h.cache.reset()
        rows[label] = _rows(h)
        if args.rows_only:
            r = rows[label]
            print(
                f"{label:<24} real {r['real']:.3f}  spec {r['spec']:.3f}  "
                f"total {r['total']:.3f} rows/layer  stages {r['stages']:.2f}"
            )
            continue
        graphs[label] = h.capture()
    if args.rows_only:
        # The model's measured baseline is 2.742 real rows per layer with prefetch off;
        # a harness far from that is measuring a different cache regime entirely.
        print("\n(production reference: 2.742 real rows/layer with prefetch off)")
        return

    # Interleaved rounds: the GPU's clocks drift over a run, so timing every config back
    # to back once and comparing would partly measure the drift. Round-robin instead and
    # take the median across rounds.
    samples: dict[str, list[float]] = {label: [] for label, *_ in configs}
    xs = _inputs(h.dev, args.reps + 5)
    for round_id in range(args.rounds + 1):
        for label, *_ in configs:
            graph, stream = graphs[label]
            with torch.cuda.stream(stream):
                for i in range(5):
                    h.hidden.copy_(xs[i])
                    graph.replay()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(args.reps):
                    h.hidden.copy_(xs[i + 5])
                    graph.replay()
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.reps * 1e3
            if round_id:  # round 0 is warm-up: clocks ramp, kernels are still cold
                samples[label].append(ms)

    def spread(xs_: list[float]) -> float:
        """Half the central 50% -- a stall in one round should not become the error bar,
        but the spread has to describe the bulk of the distribution honestly."""
        q = statistics.quantiles(xs_, n=4) if len(xs_) >= 4 else [min(xs_), 0, max(xs_)]
        return q[2] - q[0]

    # min, not mean or median, is the headline. A production server may be live on this
    # box and interference can only ever ADD time, so the fastest round is the cleanest
    # estimate of what the configuration costs; the median and IQR are printed next to it
    # so a run where the two disagree is visible rather than quietly averaged away.
    base_label = "early join (shipped)"
    best = {label: min(samples[label]) for label, *_ in configs}
    base, off = best[base_label], best["prefetch off"]
    print(
        f"\n{'configuration':<24}{'min ms':>9}{'median':>9}{'IQR':>8}{'vs shipped':>12}"
        f"{'real rows':>11}{'spec rows':>11}{'total rows':>11}"
    )
    for label, *_ in configs:
        delta = "" if label == base_label else f"{(base - best[label]) / base * 100:+6.2f}%"
        r = rows[label]
        print(
            f"{label:<24}{best[label]:9.3f}{statistics.median(samples[label]):9.3f}"
            f"{spread(samples[label]):8.3f}{delta:>12}"
            f"{r['real']:11.3f}{r['spec']:11.3f}{r['total']:11.3f}"
        )

    # A delta smaller than the run-to-run spread of the minima is not a result, and
    # saying so is the point of printing it.
    noise = max(sorted(s)[1] - min(s) for s in samples.values())
    print(
        f"\nnoise floor (widest gap from best to second-best round): {noise:.3f} ms/step "
        f"= {noise / base * 100:.2f}% of the shipped baseline; "
        f"{args.rounds} rounds x {args.reps} replays"
    )
    serial = best.get("pull on main (A/B)")
    print(f"prefetch off -> shipped early join: {(off - base) / off * 100:+.2f}%")
    if serial:
        print(
            f"the fork itself (same bytes, main stream -> branch): "
            f"{(serial - base) / serial * 100:+.2f}%"
        )
    for label, *_ in configs:
        if label in (base_label, "prefetch off"):
            continue
        gain = (base - best[label]) / base * 100
        verdict = "WITHIN NOISE" if abs(base - best[label]) < noise else "outside noise"
        print(f"{label:<24} vs shipped: {gain:+6.2f}%  ({verdict})")


if __name__ == "__main__":
    main()
