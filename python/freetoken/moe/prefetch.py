"""In-graph L+1 MoE expert prefetch -- env-gated, OFF by default.

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

    MAIN    ... ensure(L) copy(L) | Gate_{L+1}(x_L) prefetch_ensure(L+1) [fork] ... GEMM(L)
    BRANCH                                                    \\-> prefetch_copy(L+1) -[done]

and the branch's completion event is joined on the main stream before layer L+1's
expert GEMM (see ``join``), one layer later.

Ordering rules the implementation preserves:

* every ``lru_ensure`` -- real and speculative -- stays on the main stream, in
  program order. Only the pull forks, so the slot tables are never written
  concurrently.
* the real ``copy_missing(L)`` still completes before ``GEMM(L)`` (untouched).
* the prefetch pull for L+1 completes before ``GEMM(L+1)`` (the join).
* the real path's single-buffered miss plan is never read or written by the
  prefetch path -- that is what the second descriptor set on the cache is for.

Keep-wrong semantics
--------------------
A mispredicted admission stays in the cache as an ordinary LRU entry. There is no
staging area and no pinning: an entry admitted this step is the freshest thing in
the cache, so the ordinary victim search will not take it back for many steps, and
offline replay of the trace showed keep-wrong and discard-wrong worth the same.

The two eviction hazards, and why neither is left to luck
--------------------------------------------------------
Speculation shares the LRU state with the real path, so a speculative admission can
evict something. Two cases matter, and both are closed structurally rather than by
hoping the cache is big enough (an early version relied on recency alone and DID
corrupt outputs once the toy cache was thrashed hard enough -- the race is real, and
it is timing dependent, which is the worst kind to ship on a hope):

1. ``prefetch_ensure(L+1)`` runs while ``GEMM(L)`` is still queued behind it on the
   main stream, and the forked pull then writes the slots it admitted. Evicting a
   slot ``GEMM(L)`` is about to read would overwrite a row mid-GEMM.
   *Closed by* ``cache.protect_slots(topk_ids)`` immediately before the speculative
   ensure: ``lru_ensure`` refuses to evict slots whose usage equals the step it is
   about to compute, so writing that value into layer L's routed slots takes them out
   of the victim pool for exactly that one call.
2. ``ensure_experts(L+1)`` one layer later could evict a slot the branch pull is
   still writing, after which ``copy_missing(L+1)`` writes the same slot from the
   main stream -- two writers, one row.
   *Closed by* joining the branch BEFORE layer L+1's real ensure, so the pull has
   retired before any of L+1's bookkeeping runs.

Where the join goes, and why not later
--------------------------------------
The original design called for the join just before layer L+1's expert GEMM rather
than before its ensure, for a wider overlap window. That is unsound as built, for a
reason beyond hazard 2: the PREFETCH plan is single-buffered exactly like the real
one. Deferring the join past ``schedule(L+2)`` lets ``prefetch_ensure(L+2)`` overwrite
``prefetch_evict_slots``/``prefetch_num_indices`` while the branch's pull for L+1 is
still reading them, so the pull lands wrong rows in wrong slots. It reproduced: with
an exact lookahead -- the case where the GEMM's rows come from the branch rather than
from the main-stream copy -- a late join failed the value-identity test in 4 of 4
runs under load, while joining before the ensure never failed in any configuration.

Joining before the ensure gives up very little: only the overlap against
``ensure_experts(L+1)`` (one small kernel) and ``copy_missing(L+1)``, which is itself
a PCIe pull the prefetch would merely have contended with on the same link. Restoring
the later join would require parity-indexed double buffering of the prefetch plan; it
is not worth that for an overlap window with no bandwidth to gain.

Independently of both hazards, ``cache_size >= 4 * K`` (K = predicted ids =
batch x top_k) is required and enforced: the victim pool must still contain K
evictable slots after the protected ones are removed, and ``lru_ensure`` has no
defined behaviour when it cannot find enough. Layers failing the bound are skipped,
once-logged -- the same decision on every step, so it stays capture-safe.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

__all__ = ["MoePrefetcher", "get_prefetcher"]


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
        # Layers with a pull in flight for the CURRENT forward. Host-side bookkeeping
        # evaluated while the graph is traced, never during replay; the entry is
        # popped by the join so an aborted forward cannot leave a stale one.
        self._armed: dict[int, bool] = {}
        self._skipped: set[int] = set()

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
    def _eligible(self, cache, src: int, dst: int) -> bool:
        """Whether layer ``src`` may prefetch layer ``dst``'s experts.

        Every term is a host-side, capture-time constant (registry membership,
        cache configuration), so the answer is identical on every step of a
        captured graph -- which is what makes the fork edges static.
        """
        if src not in self._gates or dst not in self._gates:
            return False
        if cache is None or cache.prefetch_stream is None:
            return False
        # hybrid caps the real fetch on purpose (the CPU absorbs the overflow); an
        # uncapped speculative pull would spend exactly the PCIe budget that cap
        # exists to protect. CPU layers never touch the slot cache at all.
        if cache.decode_target != "gpu":
            return False
        if cache.is_cpu_layer(src) or cache.is_cpu_layer(dst):
            return False
        if cache.is_unpinned_layer(dst):
            return False
        # The copy-engine doorbell owns the main stream's copy protocol end to end
        # (staged mirrors + a spin kernel); it has no second-plan equivalent.
        if cache.dma_service is not None:
            return False
        # blocks_per_bank -- the whole reason the forked pull overlaps at all --
        # only exists on the fused multi-bank kernel.
        if not cache._copy_fused_ok:
            return False
        return True

    def prefetches(self, cache, layer_id: int) -> bool:
        """Whether ``layer_id`` schedules a prefetch of ``layer_id + 1``."""
        return self._eligible(cache, layer_id, layer_id + 1)

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
        dst = layer_id + 1
        if not self._eligible(cache, layer_id, dst):
            return
        if hidden is None or input_ids is None:
            return

        # Lookahead router -- MAIN stream, serialized with every other ensure.
        _weights, predicted = self._gates[dst](hidden, input_ids)
        k = predicted.numel()
        if cache.cache_size < 4 * k:
            self._skip_once(
                dst,
                f"cache_size {cache.cache_size} < 4 * {k} predicted ids: too few "
                "evictable slots left once the current layer's rows are protected",
            )
            return
        # Hazard 1 (see the module docstring): take this layer's live rows out of the
        # speculative ensure's victim pool.
        cache.protect_slots(active_slots)
        cache.prefetch_ensure(dst, predicted)

        main = torch.cuda.current_stream(cache.device)
        branch = cache.prefetch_stream
        fork_event = cache.prefetch_fork_events[dst]
        done_event = cache.prefetch_done_events[dst]
        # Fork: the branch starts only after the speculative ensure has produced the
        # plan it is about to read.
        fork_event.record(main)
        branch.wait_event(fork_event)
        with torch.cuda.stream(branch):
            cache.prefetch_copy()
        done_event.record(branch)
        self._armed[dst] = True

    def join(self, cache, layer_id: int) -> None:
        """Order this layer's MoE block behind the pull scheduled for it one layer ago.

        Called immediately before the layer's real ``ensure_experts`` -- see the module
        docstring for why it cannot be deferred to just before the expert GEMM.
        """
        if not self._armed.pop(layer_id, False):
            return
        torch.cuda.current_stream(cache.device).wait_event(
            cache.prefetch_done_events[layer_id]
        )


_PREFETCHER: MoePrefetcher | None = None


def get_prefetcher() -> MoePrefetcher:
    """The process-wide prefetcher. Constructed lazily (no CUDA, no tensors) so the
    CLI flag can still be mapped to the environment after freetoken is imported."""
    global _PREFETCHER
    if _PREFETCHER is None:
        _PREFETCHER = MoePrefetcher()
    return _PREFETCHER
