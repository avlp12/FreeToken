"""Second (prefetch) miss-plan descriptor set: a speculative admission must admit
its ids into the shared LRU state and write its plan into the PREFETCH descriptors,
leaving the real path's single-buffered plan byte-for-byte untouched.

The real decode path stages evict_slots/src_indices/num_indices and consumes them in
the same layer, so anything that could overwrite them between ``ensure_experts`` and
``copy_missing`` would silently fetch the wrong expert rows.
"""

from __future__ import annotations

import os

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FEATS = [4096, 2048]  # bf16 schema (gate_up, down); 16B-aligned so the fused plan builds


def _build_cache(num_layers=4, num_experts=8, cache_size=64, prefetch=True):
    from freetoken.moe.offload_cache import MOE_PREFETCH_ENV, OffloadMoeCache

    prev = os.environ.get(MOE_PREFETCH_ENV)
    os.environ[MOE_PREFETCH_ENV] = "1" if prefetch else "0"
    try:
        dev = torch.device("cuda")
        cache = OffloadMoeCache(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            device=dev,
            prefill_overlap=False,
            quant_format="bf16",
        )
    finally:
        if prev is None:
            os.environ.pop(MOE_PREFETCH_ENV, None)
        else:
            os.environ[MOE_PREFETCH_ENV] = prev
    sources = {
        name: list(
            torch.randint(
                0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev
            )
            .view(torch.bfloat16)
            .split(num_experts)
        )
        for name, feat in zip(("gate_up", "down"), FEATS)
    }
    cache.set_bank_sources(sources)
    return cache


@CUDA
def test_disabled_allocates_nothing():
    cache = _build_cache(prefetch=False)
    assert not cache.prefetch_enabled
    assert cache.prefetch_evict_slots is None
    assert cache.prefetch_src_indices is None
    assert cache.prefetch_num_indices is None
    assert cache.prefetch_stream is None
    assert cache.prefetch_fork_events == [] and cache.prefetch_done_events == []


@CUDA
def test_prefetch_plan_is_independent_of_the_real_plan():
    cache = _build_cache()
    assert cache.prefetch_enabled and cache._copy_fused_ok
    dev = torch.device("cuda")

    # Real path for layer 0: stage a plan, but do NOT consume it yet.
    real_ids = torch.tensor([[1, 3, 5]], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, real_ids)
    real_n = int(cache.num_indices.item())
    real_plan = (cache.evict_slots[:real_n].clone(), cache.src_indices[:real_n].clone())
    assert real_n == 3  # cold cache: every routed expert is a miss

    # Speculative admission for layer 1 lands in the SECOND descriptor set.
    pred = torch.tensor([[2, 4, 6, 2]], dtype=torch.int64, device=dev)  # dup -> one copy
    cache.prefetch_ensure(1, pred)
    torch.cuda.synchronize()

    assert int(cache.prefetch_num_indices.item()) == 3
    # The real plan survived untouched.
    assert int(cache.num_indices.item()) == real_n
    assert torch.equal(cache.evict_slots[:real_n], real_plan[0])
    assert torch.equal(cache.src_indices[:real_n], real_plan[1])
    # The prefetch plan points at layer-1 rows, in disjoint slots.
    p_n = int(cache.prefetch_num_indices.item())
    assert sorted(cache.prefetch_src_indices[:p_n].tolist()) == [2, 4, 6]
    assert set(cache.prefetch_evict_slots[:p_n].tolist()).isdisjoint(
        real_plan[0].tolist()
    )
    # Shared LRU state sees the speculative ids as resident.
    assert (cache.slot_for_id[1, [2, 4, 6]] >= 0).all()
    # The predicted-id tensor the caller passed is NOT rewritten in place.
    assert pred.flatten().tolist() == [2, 4, 6, 2]
    # Speculation must not pollute the real hit/miss counters.
    assert int(cache.lru_stats.sum()) == 0


@CUDA
def test_prefetch_copy_moves_the_predicted_rows():
    cache = _build_cache()
    dev = torch.device("cuda")
    for _, c in cache.banks:
        c.zero_()

    pred = torch.tensor([[2, 4, 6]], dtype=torch.int32, device=dev)
    cache.prefetch_ensure(2, pred)
    cache.prefetch_copy()
    torch.cuda.synchronize()

    n = int(cache.prefetch_num_indices.item())
    slots = cache.prefetch_evict_slots[:n].tolist()
    rows = cache.prefetch_src_indices[:n].tolist()
    # bitwise: the random bf16 payload contains NaNs, so compare the raw bytes.
    for per_layer, cachetensor in cache.banks:
        for slot, row in zip(slots, rows):
            assert torch.equal(
                cachetensor[slot].view(torch.int16), per_layer[2][row].view(torch.int16)
            )


@CUDA
def test_real_ensure_sees_a_correct_prediction_as_a_hit():
    cache = _build_cache()
    cache.collect_stats = True
    dev = torch.device("cuda")

    pred = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=dev)
    cache.prefetch_ensure(1, pred)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=dev)
    slots_expected = cache.slot_for_id[1, [1, 2, 3]].clone()
    cache.ensure_experts(1, ids)
    torch.cuda.synchronize()

    assert int(cache.num_indices.item()) == 0, "prefetched experts must be hits"
    assert torch.equal(ids.flatten(), slots_expected)


@CUDA
def test_protect_slots_survives_the_next_ensure():
    """``protect_slots`` must take the current layer's live rows out of the NEXT
    lru_ensure's victim pool, even when they are the only plausible victims.

    Without it, the speculative admission for L+1 can evict a slot layer L's expert
    GEMM -- still queued behind it on the main stream -- is about to read, and the
    forked pull then overwrites that row mid-GEMM.
    """
    # 8 slots, 6 live rows: every admission below MUST come out of the other 2.
    cache = _build_cache(num_layers=4, num_experts=8, cache_size=8)
    dev = torch.device("cuda")

    live = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, live)  # -> slot ids, in place
    live_slots = sorted(live.tolist())

    cache.protect_slots(live)
    pred = torch.tensor([0, 1], dtype=torch.int32, device=dev)  # layer 1, both missing
    cache.prefetch_ensure(1, pred)
    torch.cuda.synchronize()

    n = int(cache.prefetch_num_indices.item())
    assert n == 2
    victims = cache.prefetch_evict_slots[:n].tolist()
    assert set(victims).isdisjoint(live_slots), (victims, live_slots)
    # ... and layer 0's rows are still mapped where its GEMM expects them.
    assert sorted(cache.slot_for_id[0, :6].tolist()) == live_slots


@CUDA
def test_unprotected_slots_are_evictable():
    """Control for the test above: without protect_slots the same admission DOES take
    the live rows (which is exactly the race the protection exists to remove)."""
    cache = _build_cache(num_layers=4, num_experts=8, cache_size=8)
    dev = torch.device("cuda")
    live = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, live)  # fills every slot
    live_slots = set(live.tolist())

    pred = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    cache.prefetch_ensure(1, pred)
    torch.cuda.synchronize()
    victims = cache.prefetch_evict_slots[:2].tolist()
    assert set(victims) <= live_slots, "nothing else could have been evicted"


@CUDA
def test_rebuild_reallocates_the_prefetch_plan():
    cache = _build_cache(cache_size=64)
    stream, events = cache.prefetch_stream, cache.prefetch_done_events
    cache.rebuild(128)
    assert cache.prefetch_evict_slots.numel() == 128
    assert cache.prefetch_src_indices.numel() == 128
    assert int(cache.prefetch_num_indices.item()) == 0
    # Handles a captured graph may reference are kept.
    assert cache.prefetch_stream is stream and cache.prefetch_done_events is events
