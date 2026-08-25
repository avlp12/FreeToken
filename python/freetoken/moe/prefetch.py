"""In-graph L+1 / L+2 MoE expert prefetch -- env-gated, OFF by default.

Enabled by ``FREETOKEN_MOE_PREFETCH=1`` (or ``--moe-prefetch``, which sets it).
With the flag unset the cache allocates no prefetch state, every offload MoE layer
holds ``prefetcher is None``, and the captured decode graph is the current one
kernel for kernel.

The idea
--------
Decode is PCIe bound: ~26ms of the ~47.5ms token budget is expert rows crossing the
link, serialized in front of each layer's GEMM (``ensure_experts`` -> ``copy_missing``
-> GEMM, all on the main stream). The rows for layer L+1 cannot start moving until
layer L+1's router has run -- unless something predicts them earlier.

Layer L's router input, pushed through layer L+1's OWN Gate, predicts layer L+1's
actual top-6 at 71.5% recall on this model (measured with the routing tracer, see
``freetoken/moe/routing_trace.py``); the leading hash-router layers are predicted
exactly, since their Gate is a token-id lookup that does not depend on the hidden
state at all. That is enough lead time to move most of layer L+1's rows while layer
L is still computing.

What runs where
---------------
Per decode step, at every MoE layer L whose successor is also an offloaded MoE
layer, after L's REAL ``ensure_experts``/``copy_missing`` and before L's expert GEMM::

    MAIN  guard(L) ensure(L) copy(L) | Gate_{L+1}(x_L) pf_ensure(L+1)
                                       Gate_{L+2}(x_L) pf_ensure(L+2) [fork] join(L) GEMM(L)
    BRANCH                                      \\-> pf_copy(L+1) pf_copy(L+2) -[done_L]

with ``join(L)`` waiting on the done event layer ``L-1`` recorded -- so the window a
pull has to hide in runs from layer L-1's fork all the way to layer L's GEMM.

Ordering rules the implementation preserves:

* every ``lru_ensure`` -- real and speculative -- stays on the main stream, in
  program order. Only the pull forks, so the slot tables are never written
  concurrently.
* the real ``copy_missing(L)`` still completes before ``GEMM(L)`` (untouched).
* the prefetch pull for L+1 completes before ``GEMM(L+1)`` (the join). The L+2 pull is
  covered transitively: it is queued on the same branch ahead of layer L+1's pulls, so
  the join at L+2 (which waits on layer L+1's done event) implies it.
* the real path's single-buffered miss plan is never read or written by the
  prefetch path -- that is what the separate descriptor sets on the cache are for.
* a descriptor set is only rewritten after every pull that reads it has retired --
  that is what the PARITY index on those sets is for (see below).

The second lookahead stage (L+2)
--------------------------------
The same hidden that predicts L+1 at 71.5% recall predicts L+2 at 65.3% (measured,
routing trace v2). Pulling for L+2 from layer L gives a pull ~2.2ms of window instead
of ~1.1ms, and -- more usefully on a link that arrives in bursts -- lets a layer whose
L+1 prediction was wrong still have been covered two layers earlier.

But those bytes cross the SAME saturated link, and the rank-limit table below is the
lesson about what that costs. So the L+2 stage speculates NARROW: ``top-1`` by
predicted score (``FREETOKEN_PREFETCH_L2_TOPK``, 0 disables the stage entirely),
against ``top-3`` for L+1. Hash-router targets are exact and never truncated, at
either distance.

**And measured at the server, even top-1 does not pay: the stage is OFF by default.**
The narrow L+2 pull turns out to predict rows that are already resident. From
``--moe-collect-stats`` over a 399-token decode at 740 slots, L2 on vs off::

                   missing/layer   pf_pulled/layer   total rows/layer
    L2 top-1           1.691           1.251              2.943
    L2 off             1.709           1.218              2.927

It removes 0.018 real misses per layer and pulls 0.033 extra rows to do it -- it moves
MORE bytes than it saves, on the resource that is the bottleneck. What it also costs
is on the critical path: 40 extra speculative ensures, 40 extra ``protect_slots``
kernels and 40 extra pull launches per step (~55% of which retire empty, num_indices
0), and -- because a source layer records ONE done event after BOTH its stages --
``join(L+1)`` waits on the L+2 pull as well as the L+1 pull it actually needs.

A/B at the server, 300-step CUDA-event decode timing, five interleaved pairs across
both serving configs (740-slot/320K and 1421-slot/128K). Every pair favours OFF, on
mean, on median and on end-to-end tok/s; median deltas -1.3% to -4.4%::

    320K  median ms   35.85 -> 34.27 | 36.82 -> 35.32 | 36.33 -> 35.32
    128K  median ms   29.95 -> 28.99 | 29.65 -> 29.27

Interleaved because the box drifts: two runs of the SAME config an hour apart differed
by 4.6%, which is the size of the effect, so a single A/B could not have carried this.

Double pulling is not a hazard the code has to defend against: the two stages predict
DIFFERENT layers (L+1 and L+2), whose rows live in disjoint id spaces, so one layer's
pair of admissions can never name the same row twice. Across layers, the row layer L
admitted for L+2 is simply resident when layer L+1 stages its own L+2 prediction --
``lru_ensure`` sees a hit and does not re-pull it. The only waste left is a row admitted
at L for L+2 and evicted before L+1 runs, which is then re-pulled; the protections below
make that rare rather than impossible, and it costs bytes, never correctness.

Why a rank limit (FREETOKEN_PREFETCH_TOPK) exists
-------------------------------------------------
Decode here is **bandwidth** bound, not latency bound, and that changes what the
feature is for. Measured on this box -- DSV4-Flash q2_k_ud, 740 slots, 399-token
decode, expert rows 9.19 MiB, 43 layers, from ``moe-stats``::

                     real miss   spec     total rows   sustained
                     /layer      pulled   /layer       PCIe        tok/s
    off              2.742       --       2.742        24.2 GB/s   21.3
    top-2, bpb 4     1.955       0.862    2.818        27.3 GB/s   23.4
    top-3, bpb 4     1.689       1.226    2.915        29.2 GB/s   24.2   <- default
    top-4, bpb 4     1.434       1.647    3.081        30.9 GB/s   24.2
    all 6,  bpb 2    1.080       2.691    3.771        31.9 GB/s   20.4

(bytes/token = 43 layers x rows x 9.19 MiB; GB/s = that over the measured step.)

Read down the columns and the whole mechanism is visible. The lookahead genuinely
predicts: every extra rank cuts real misses further, and at all six it removes 61% of
them. The fork genuinely overlaps: every extra rank also lifts sustained PCIe, from
24.2 GB/s (the serial path leaves a quarter of the link idle behind compute) toward
31.9 GB/s.

(The "31.9 GB/s is this platform's zero-copy ceiling" this section used to claim is
wrong, and was wrong in a direction that made the feature look boxed in. Per-kernel
timings from a CUPTI trace of the replayed decode graph put an UNCONTENDED
``fast_index_copy_multi`` at 50.4 GB/s on the serial path (bpb 8) and 44.5 GB/s on the
forked pull (bpb 4) -- ~80% of PCIe Gen5 x16. Sustained over the whole step decode
reaches only 35.2 GB/s, and over just the copy-active part of the step 46.3 GB/s. So
the link is NOT saturated: it is idle for 24% of the step. See the accounting below.)

But every rank costs bytes on a link that is the bottleneck, and the low-ranked
predictions are the ones most likely to be wrong. At all six, 37.5% more bytes move
and the fork wins back 32% more bandwidth -- they cancel exactly, which is why the
first server measurement of this feature showed no effect at all despite both halves
working. Only ~1.66 of those 2.691 speculative rows were rows the real path would
have fetched anyway; the rest is waste billed to the exhausted resource.

Half the router's top_k is the measured optimum and the default (an explicit
``FREETOKEN_PREFETCH_TOPK`` overrides; 0 pulls every prediction). Hash-router layers
are never truncated: their Gate is a token-id lookup, exact by construction, so every
row it names would have been fetched anyway -- and its ids carry no score order to
truncate by.

What decode's copy time actually decomposes into
------------------------------------------------
A CUPTI trace of 20 replayed decode steps (``FREETOKEN_DECODE_PROFILE``, 740 slots)
splits cleanly, because the two copy paths compile to distinguishable kernels: the
serial ``copy_missing`` is ``fast_index_copy_multi<int,1024,8>`` (``copy_bpb``) and the
forked pull is ``...<int,1024,4>`` (``prefetch_bpb``). Wall time by which classes are
live, per step (GPU busy 34.64 ms; CUDA-event step time on the same config 34.88 ms,
so this closes to 0.7%)::

    compute alone (link idle)                              8.33 ms   24.0%
    compute || forked pull                                 7.40 ms   21.4%
    serial copy_missing alone                             11.34 ms   32.7%
    serial copy_missing || forked pull                     4.28 ms   12.4%
    forked pull alone (main stream stalled at the join)    3.29 ms    9.5%
    serial copy_missing || compute                         0.00 ms    0.0%

That last row is the one that reframes the feature. ``copy_missing(L)`` is the
dependency ``GEMM(L)`` waits on, on the same stream, so it can never overlap compute --
measured 0.000 ms, not "poorly overlapped". Of 30.60 ms of copy kernel time per step,
15.63 ms is structurally unhideable and 14.97 ms (the pull) is the only part that CAN
hide. So "the copy stream overlaps compute 24% of the time" is not a quarter of the
achievable: the ceiling is 49%, the prefetch share, and 49.4% of the pull is in fact
already hidden. The feature is at half its ceiling, not a quarter.

Visible copy = GPU busy - compute = 18.91 ms/step, and it decomposes as::

    serial real misses (34.4 real copies/step x 403.7us uncontended)   13.89 ms
    unhidden prefetch (pull work at solo rate, minus what hid)          4.24 ms
    copy-vs-copy contention, net                                        0.78 ms

The contention term is small and that is a result, not an omission: running the two
copy classes concurrently inflates their kernel time by 5.06 ms (a contended serial
copy is 632.9us against 403.7us solo, x1.57; a contended pull 708.5us against 310.0us,
x2.29) but buys back 4.28 ms of wall. The link arbitrates close to work-conservingly,
so pulling harder on it is not where the win is -- confirmed by sweeping the pull's
grid, where ``prefetch_bpb`` 4 beats both 2 (+6.1%) and 8 (+1.8%).

The 4.24 ms of unhidden prefetch is a LEAD-TIME shortfall, not a scheduling bug:

* it is concentrated -- the top 8 of 43 layers hold 77.6% of the join stall, and per
  layer the stall tracks the size of the pull forked one layer earlier;
* it alternates with the model's own structure, ``compress_ratios`` running 4/128/4/128
  from layer 2, so consecutive layers offer very different amounts of compute for the
  same pull to hide behind;
* the hash layers are the extreme case. ``n_hash_layers=3``, and their Gate is exact,
  so layers 1 and 2 have NO real misses at all (their ``copy_missing`` retires in ~1us)
  and every row they need arrives speculatively -- a 1193us pull against a one-layer
  window, which stalls 1948us. Perfect prediction does not help when there is only one
  layer of lead time to spend it in.

And the 8.33 ms of link idle is spread evenly, ~190us per layer, rather than pooled
anywhere: it is the interval in each layer during which no copy is KNOWABLE -- the
one-layer-lead pull has drained and the next layer's real routing has not run yet.
Closing it needs more lead, not a different fork/join schedule; see the note on
depth-D speculation in the design docs before reaching for one.

Keep-wrong semantics
--------------------
A mispredicted admission stays in the cache as an ordinary LRU entry. There is no
staging area and no pinning: an entry admitted this step is the freshest thing in
the cache, so the ordinary victim search will not take it back for many steps, and
offline replay of the trace showed keep-wrong and discard-wrong worth the same.

Where the join goes: just before GEMM(L)
---------------------------------------
The join used to sit before layer L's real ``ensure_experts``. That was sound but
short: it gave up the overlap against ``ensure_experts(L)``, ``copy_missing(L)`` and
both lookahead Gates. It sat there because the prefetch plan was single-buffered
exactly like the real one, so deferring the join past the next layer's staging let
``prefetch_ensure`` overwrite ``evict_slots``/``num_indices`` while the branch's pull
was still reading them -- the pull then lands wrong rows in wrong slots. That
reproduced hard: a naive late join failed value identity in 4 of 4 runs under load.

The server confirms the placement is worth the machinery it needs: at 740 slots the
early join costs 5.5% (36.48 -> 38.49 ms mean per decode step), and at 1421 slots the
two are within noise of each other (30.46 vs 30.14). So late stays the default.

Two structural changes move the join to just before the expert GEMM.

**Parity-indexed plans.** The descriptor sets are indexed ``(src_layer % 2, stage)``
(``OffloadMoeCache.prefetch_plan_index``). Layer L's plans are overwritten by layer
L+2, and layer L+1's join -- which precedes layer L+1's GEMM, hence everything layer
L+2 does -- has already waited on layer L's done event. So no descriptor is ever
rewritten while a pull reading it is in flight. ``MoePrefetcher.double_buffer = False``
collapses the parity and restores the racy single-buffered layout, which is how the
test proves the race is what the buffering fixes.

**Re-protecting the in-flight pull's destinations.** With the join late, layer L+1's
real ``ensure_experts`` now runs while layer L's pull is going. See hazard 2.

The three eviction hazards, and why none is left to luck
-------------------------------------------------------
Speculation shares the LRU state with the real path, so a speculative admission can
evict something. ``lru_ensure`` bumps ``lru_step`` on entry and refuses to evict slots
whose ``usage`` equals the step it is about to compute; ``protect_slots`` writes that
value, which shields the named rows from exactly ONE following call. Every protection
below is therefore issued immediately before the call it must survive -- an early
version relied on recency alone and DID corrupt outputs once the toy cache was thrashed
hard enough. The race is real, and it is timing dependent, which is the worst kind to
ship on a hope.

1. **Speculation evicting the current GEMM's rows.** ``prefetch_ensure`` runs while
   ``GEMM(L)`` is still queued behind it on the main stream, and the forked pull then
   writes the slots it admitted. Evicting a slot ``GEMM(L)`` is about to read would
   overwrite a row mid-GEMM.
   *Closed by* ``cache.protect_slots(topk_ids)`` before EACH lookahead ensure -- both
   of them, since one protection covers one call.
2. **The real path evicting an in-flight pull's destinations.** ``ensure_experts(L+1)``
   now runs before the join, so it can pick a slot layer L's pull is still writing;
   ``copy_missing(L+1)`` would then write the same row from the main stream. Two
   writers, two streams, one row.
   *Closed by* ``cache.protect_prefetch_plan`` on both of layer L's plan sets,
   immediately before ``ensure_experts(L+1)`` (``MoePrefetcher.guard``). Only layer L's
   fork can be in flight there: everything layer L-1 forked was joined at layer L.
3. **The L+2 stage evicting what the L+1 stage just admitted.** The two lookahead
   ensures are consecutive calls, so the second could take back the first's rows.
   *Closed by* ``protect_prefetch_plan`` on the stage-0 plan before the stage-1 ensure.

A speculative admission evicted LATER -- by layer L+1's own lookahead ensures, after
the guard's one-call protection has lapsed -- is deliberately left alone: the only
other writer for such a row is a later pull on the SAME branch stream, which is ordered
behind the one in flight, so the last write and the slot table agree.

Independently of all three, ``cache_size >= 4 * K`` (K = every predicted id this layer
stages, both stages together) is required and enforced: the victim pool must still
contain K evictable slots after the protected ones are removed, and ``lru_ensure`` has
no defined behaviour when it cannot find enough. A layer that fails the bound drops the
L+2 stage first and skips entirely only if L+1 alone will not fit -- once-logged, and
the same decision on every step, so it stays capture-safe.
"""

from __future__ import annotations

import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

__all__ = ["MoePrefetcher", "get_prefetcher"]

# How many of the lookahead router's top-k predictions to actually pull, per layer.
# Unset = auto (half the router's top_k, the measured optimum -- see the "Why a rank
# limit" section of the module docstring); 0 = all of them; N = that many. On a
# bandwidth-saturated link the low-ranked predictions cost more in wasted bytes than
# the misses they remove, so this is the knob that decides whether the feature is a
# win or a wash.
TOPK_ENV = "FREETOKEN_PREFETCH_TOPK"
# How many of the L+2 lookahead's predictions to pull, per layer. Default 0: OFF --
# no second Gate call, no second ensure, no second pull, and the captured graph is the
# L+1-only one kernel for kernel. The stage was designed to cover what the L+1 stage
# misses and to smooth bursts, but at the server the rows it names are already
# resident: it removes 0.018 real misses per layer while pulling 0.033 extra rows, and
# its pull sits on the critical path because the source layer's single done event
# makes join(L+1) wait for it. Five interleaved A/B pairs across both serving configs
# all favour off. See "The second lookahead stage (L+2)" in the module docstring.
# Set to 1 (or more) to restore it.
L2_TOPK_ENV = "FREETOKEN_PREFETCH_L2_TOPK"
L2_TOPK_DEFAULT = 0
# Where the branch join goes. 1 (default) = just before the expert GEMM, the wide
# window the parity-indexed descriptors make sound. 0 = the old placement, before the
# layer's real ensure_experts. Kept as a knob purely so the placement can be A/B'd at
# the server without a rebuild; both placements are correct.
LATE_JOIN_ENV = "FREETOKEN_PREFETCH_LATE_JOIN"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


class MoePrefetcher:
    """Process-wide lookahead-Gate registry plus the fork/join orchestration.

    Gates register at model construction (unconditionally -- a dict entry costs
    nothing and keeps the registry independent of when the env flag is read); the
    orchestration only ever runs from a layer that holds a non-None ``prefetcher``,
    which ``attach_offload_moe_cache`` sets exactly when the cache has prefetch
    state allocated.
    """

    def __init__(self) -> None:
        self._gates: dict[int, torch.nn.Module] = {}
        self.n_layers = 0
        self.top_k = 0
        self.n_experts = 0
        # Layers that must join a pull before their GEMM, mapped to
        # (source layer that forked it, how many stages it forked). Host-side
        # bookkeeping evaluated while the graph is traced, never during replay; the
        # entry is popped by the join so an aborted forward cannot leave a stale one.
        self._armed: dict[int, tuple[int, int]] = {}
        self._skipped: set[int] = set()
        # Layers a fork was actually enqueued for, ever. Under CUDA graphs `schedule`
        # only runs while the graph is being traced, so a non-empty set is proof the
        # fork/join nodes are IN the captured decode graph -- the one thing a server
        # log otherwise cannot tell you.
        self.scheduled: set[int] = set()
        raw = os.getenv(TOPK_ENV, "").strip()
        self._topk_env = int(raw) if raw else None  # None -> auto, from top_k
        # 0 = pull every prediction. Replaced with the resolved limit by the first
        # register_gate, which is where the router's top_k becomes known.
        self.topk_limit = 0
        # Second lookahead stage (L+2). 0 = the stage is off and nothing about the
        # L+1-only graph changes.
        self.l2_topk = max(0, _env_int(L2_TOPK_ENV, L2_TOPK_DEFAULT))
        # Parity-indexed descriptor sets; False restores the racy single-buffered
        # layout (tests only -- see the module docstring).
        self.double_buffer = True
        # Join before the expert GEMM (True) or before the layer's real ensure (False).
        self.late_join = _env_int(LATE_JOIN_ENV, 1) != 0

    # ------------------------------------------------------------------ registry
    def register_gate(
        self,
        layer_id: int,
        gate: torch.nn.Module,
        *,
        n_layers: int,
        top_k: int,
        n_experts: int,
    ) -> None:
        """Publish layer ``layer_id``'s router so layer ``L-1`` can run it ahead.

        Layer ids at or beyond ``n_layers`` (a speculative drafter continues the
        target's id space) are ignored: prefetching across the target/drafter
        boundary would predict with the wrong stack's router.
        """
        if layer_id >= n_layers:
            return
        if self.n_layers == 0:
            self.n_layers, self.top_k, self.n_experts = n_layers, top_k, n_experts
            # Auto rank limit: half the router's top_k. Measured optimum at top_k=6
            # (see the module docstring's table); expressed as a fraction so a model
            # with a different top_k lands somewhere sane rather than on a literal 3.
            self.topk_limit = (
                self._topk_env if self._topk_env is not None else max(1, top_k // 2)
            )
        elif (self.n_layers, self.top_k, self.n_experts) != (n_layers, top_k, n_experts):
            # Mixed geometries in one process: keep the first and prefetch nothing
            # for the rest rather than mispredicting into a foreign id space.
            logger.warning(
                "MoE prefetch: mixed MoE geometries in one process; "
                f"layer {layer_id} not registered"
            )
            return
        self._gates[layer_id] = gate

    # ------------------------------------------------------------- eligibility
    def ineligible_reason(self, cache, src: int, dst: int) -> str | None:
        """Why layer ``src`` may not prefetch layer ``dst``, or None if it may.

        Every term is a host-side, capture-time constant (registry membership,
        cache configuration), so the answer is identical on every step of a
        captured graph -- which is what makes the fork edges static, and what lets
        :meth:`describe` report the whole picture at attach time, before a single
        decode has run.
        """
        if src not in self._gates:
            return f"layer {src} registered no lookahead Gate"
        if dst not in self._gates:
            return f"layer {dst} registered no lookahead Gate (last MoE layer?)"
        if cache is None or cache.prefetch_stream is None:
            return "the cache allocated no prefetch stream (feature off, or CPU device)"
        # hybrid caps the real fetch on purpose (the CPU absorbs the overflow); an
        # uncapped speculative pull would spend exactly the PCIe budget that cap
        # exists to protect. CPU layers never touch the slot cache at all.
        if cache.decode_target != "gpu":
            return f"decode_target is {cache.decode_target!r}, not 'gpu'"
        if cache.is_cpu_layer(src) or cache.is_cpu_layer(dst):
            return "one of the two layers decodes on the CPU executor"
        if cache.is_unpinned_layer(dst):
            return f"layer {dst}'s host banks are not pinned"
        # The copy-engine doorbell owns the main stream's copy protocol end to end
        # (staged mirrors + a spin kernel); it has no second-plan equivalent.
        if cache.dma_service is not None:
            return "--moe-copy-engine owns the copy path"
        # blocks_per_bank -- the whole reason the forked pull overlaps at all --
        # only exists on the fused multi-bank kernel.
        if not cache._copy_fused_ok:
            return "the fused multi-bank copy plan is unavailable"
        return None

    def _eligible(self, cache, src: int, dst: int) -> bool:
        return self.ineligible_reason(cache, src, dst) is None

    def prefetches(self, cache, layer_id: int) -> bool:
        """Whether ``layer_id`` schedules a prefetch of ``layer_id + 1``."""
        return self._eligible(cache, layer_id, layer_id + 1)

    def stages(self, cache, layer_id: int) -> list[tuple[int, int]]:
        """``(predicted layer, rank limit)`` per lookahead stage layer ``layer_id`` runs.

        Empty when this layer prefetches nothing. Stage 0 (L+1) gates the whole schedule:
        if L+1 is not a plain GPU offload MoE layer then nothing there will ever call
        :meth:`join`, and an unjoined pull would outlive the descriptors it reads. Stage 1
        (L+2) is additive and is dropped on its own when ineligible or switched off.

        Host-side and capture-time constant, exactly like :meth:`ineligible_reason`, so
        the set of fork edges in the captured graph is fixed.
        """
        if not self._eligible(cache, layer_id, layer_id + 1):
            return []
        out = [(layer_id + 1, self.topk_limit)]
        if self.l2_topk and self._eligible(cache, layer_id, layer_id + 2):
            out.append((layer_id + 2, self.l2_topk))
        return out

    def describe(self, cache, layer_ids) -> str:
        """One-line boot summary: how many layers will prefetch, and why not the rest.

        Logged unconditionally from ``attach_offload_moe_cache`` whenever the feature
        is enabled. Silence used to be ambiguous -- no skip lines could mean "all
        good" or "nothing ever got as far as evaluating a layer" -- so the enabled
        case now always says something.
        """
        ok, l2, reasons = [], [], {}
        for layer_id in layer_ids:
            why = self.ineligible_reason(cache, layer_id, layer_id + 1)
            if why is None:
                ok.append(layer_id)
                if len(self.stages(cache, layer_id)) > 1:
                    l2.append(layer_id)
            else:
                reasons.setdefault(why, []).append(layer_id)
        join = "before GEMM" if self.late_join else "before ensure"
        parts = [
            f"{len(ok)}/{len(layer_ids)} MoE layers prefetch L+1 "
            f"(top-{self.topk_limit or 'all'}), {len(l2)} also L+2 "
            f"(top-{self.l2_topk or 'off'}), join {join}"
        ]
        for why, ids in reasons.items():
            span = f"{ids[0]}..{ids[-1]}" if len(ids) > 3 else ",".join(map(str, ids))
            parts.append(f"{len(ids)} skipped ({span}): {why}")
        return "; ".join(parts)

    def _skip_once(self, dst: int, reason: str) -> None:
        if dst not in self._skipped:
            self._skipped.add(dst)
            logger.warning(f"MoE prefetch disabled for layer {dst}: {reason}")

    # ---------------------------------------------------------- captured region
    def schedule(
        self,
        cache,
        layer_id: int,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        active_slots: torch.Tensor,
    ) -> None:
        """Predict layer ``layer_id + 1``'s experts and start pulling them.

        ``hidden`` must be the exact tensor layer ``layer_id``'s own router consumed
        (post-ffn_norm), and ``input_ids`` the matching flat token ids -- the hash
        routers key on the token, not on the hidden state. ``active_slots`` is this
        layer's ``topk_ids`` AFTER ``ensure_experts`` rewrote them to slot ids: the
        rows this layer's GEMM is about to read, which the speculative admission
        below must not be allowed to evict.

        Called on the main stream after this layer's real ``ensure_experts`` /
        ``copy_missing`` and before its expert GEMM.
        """
        stages = self.stages(cache, layer_id)
        if not stages or hidden is None or input_ids is None:
            return

        # Lookahead routers -- MAIN stream, serialized with every other ensure.
        # Rank limit: keep only the highest-scoring predictions, which are the ones
        # most likely to be right. A hash router is exact by construction (it is a
        # token-id lookup, not a function of the hidden state) and its ids carry no
        # score order, so it is never truncated -- every row it names would have been
        # fetched by the real path anyway.
        predictions = []
        for dst, limit in stages:
            gate = self._gates[dst]
            _weights, predicted = gate(hidden, input_ids)
            if limit and not getattr(gate, "hash", False):
                predicted = predicted[..., :limit]
            predictions.append((dst, predicted))

        # Victim-pool budget over EVERY id this layer stages. Drop the L+2 stage first
        # (it is the optional half); skip the layer only if L+1 alone will not fit.
        while predictions and cache.cache_size < 4 * sum(p.numel() for _, p in predictions):
            dst, predicted = predictions.pop()
            if not predictions:
                self._skip_once(
                    dst,
                    f"cache_size {cache.cache_size} < 4 * {predicted.numel()} predicted "
                    "ids: too few evictable slots left once the current layer's rows "
                    "are protected",
                )
                return
            self._skip_once(
                dst,
                f"cache_size {cache.cache_size} leaves no victim-pool budget for the "
                f"L+2 stage on top of L+1; predicting {dst} disabled",
            )

        plans = []
        for stage, (dst, predicted) in enumerate(predictions):
            plan = cache.prefetch_plan_index(layer_id, stage, self.double_buffer)
            # Hazard 1: take this layer's live rows out of this ensure's victim pool.
            # Protection lasts one call, so it is re-issued for the second stage.
            cache.protect_slots(active_slots)
            if stage:
                # Hazard 3: and keep the previous stage's fresh admissions, whose pull
                # is about to be forked, out of it too.
                cache.protect_prefetch_plan(plans[-1])
            cache.prefetch_ensure(dst, predicted, plan_index=plan)
            plans.append(plan)
            self.scheduled.add(dst)

        main = torch.cuda.current_stream(cache.device)
        branch = cache.prefetch_stream
        fork_event = cache.prefetch_fork_events[layer_id]
        done_event = cache.prefetch_done_events[layer_id]
        # Fork: the branch starts only after the speculative ensures have produced the
        # plans it is about to read. One fork and one done event per SOURCE layer,
        # covering both stages -- the branch is a single stream, so the L+2 pull is
        # ordered behind the L+1 pull and a later join subsumes both.
        fork_event.record(main)
        branch.wait_event(fork_event)
        with torch.cuda.stream(branch):
            for plan in plans:
                cache.prefetch_copy(plan_index=plan)
        done_event.record(branch)
        self._armed[layer_id + 1] = (layer_id, len(plans))

    # ------------------------------------------------------------- the two call sites
    def before_ensure(self, cache, layer_id: int) -> None:
        """Run at the top of layer ``layer_id``'s MoE block, before ``ensure_experts``.

        Late join (the default): re-protect the destination slots of the pull still in
        flight, so this layer's real ensure cannot hand one of those rows to
        ``copy_missing`` as a second writer. Early join: the join itself, which retires
        the pull outright and makes the protection moot.
        """
        if self.late_join:
            self.guard(cache, layer_id)
        else:
            self.join(cache, layer_id)

    def before_gemm(self, cache, layer_id: int) -> None:
        """Run just before layer ``layer_id``'s expert GEMM. The join, when it is late."""
        if self.late_join:
            self.join(cache, layer_id)

    def guard(self, cache, layer_id: int) -> None:
        """Shield the in-flight pull's destination rows from this layer's real ensure.

        Hazard 2 in the module docstring. Only the immediately preceding layer's fork can
        be in flight here -- everything the layer before that forked was joined one layer
        ago -- so the plan sets to protect are exactly that layer's.
        """
        armed = self._armed.get(layer_id)
        if armed is None:
            return
        src, n_stages = armed
        for stage in range(n_stages):
            cache.protect_prefetch_plan(
                cache.prefetch_plan_index(src, stage, self.double_buffer)
            )

    def join(self, cache, layer_id: int) -> None:
        """Order this layer's expert GEMM behind the pull forked for it one layer ago.

        Waits on the done event of the SOURCE layer, which covers both of that layer's
        lookahead stages (single branch stream, so they are ordered) and, transitively,
        every pull forked before it -- including the L+2 pull aimed at this layer from
        two layers back.
        """
        armed = self._armed.pop(layer_id, None)
        if armed is None:
            return
        src, _n_stages = armed
        torch.cuda.current_stream(cache.device).wait_event(cache.prefetch_done_events[src])


_PREFETCHER: MoePrefetcher | None = None


def get_prefetcher() -> MoePrefetcher:
    """The process-wide prefetcher. Constructed lazily (no CUDA, no tensors) so the
    CLI flag can still be mapped to the environment after freetoken is imported."""
    global _PREFETCHER
    if _PREFETCHER is None:
        _PREFETCHER = MoePrefetcher()
    return _PREFETCHER
