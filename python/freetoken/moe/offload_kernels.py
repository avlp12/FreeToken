from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

if TYPE_CHECKING:
    from tvm_ffi import Module

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) first fetches misses used by the most routes in THIS
# batch, then the experts most-recently active before this step (LRU on the expert).
# The route-count priority spends each paper-derived q* PCIe copy where it removes the
# most repeated CPU GEMVs during a multi-row speculative verify; recency breaks ties and
# preserves the steady-state hit-rate policy.
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)


def ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.
    """
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


def prefetch_ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """``ensure_experts`` against the SECOND miss-plan descriptor set.

    Same flashlib kernel, same shared LRU state (slot_for_id / id_of_slot / usage /
    step) -- only ``src_indices``/``dst_indices``/``num_copy`` are redirected, which
    the public ``lru_ensure`` signature already takes as caller-owned outputs. No
    flashlib change is needed and none was made.

    Stats go to ``prefetch_stats``, never ``lru_stats``: the latter measures the REAL
    routing's hit/miss rate and has to stay comparable against a prefetch-off run.
    The former is what makes the speculation itself measurable -- its MISS column is
    exactly the number of rows the forked pull will move over PCIe.
    """
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        expert_ids,
        cache.prefetch_src_indices,
        cache.prefetch_evict_slots,
        cache.prefetch_num_indices,
        stats=cache.prefetch_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


def protect_slots(cache, slots: torch.Tensor) -> None:
    """Make ``slots`` non-evictable for the NEXT ``lru_ensure`` call on this cache.

    ``lru_ensure`` bumps ``lru_step`` on entry and treats a slot as evictable exactly
    when ``usage != step`` -- i.e. only the slots THAT call hit are safe. That is not
    enough for the prefetch path: the speculative ensure for layer L+1 runs while
    layer L's expert GEMM is still queued behind it on the main stream, and if it
    evicted one of layer L's rows the forked pull would overwrite that row mid-GEMM.

    Writing ``step + 1`` -- the value the next call will compute -- into those slots'
    usage puts them in precisely the bucket that call refuses to evict, and leaves
    them ordinary (merely recent) for every call after it.

    One launch, fixed shape, no host sync -- CUDA-graph safe. The slot ids are what
    the real ``ensure_experts`` already rewrote ``topk_ids`` into, so the caller has
    them for free.
    """
    n = slots.numel()
    _protect_slots_kernel[(1,)](
        cache.usage,
        slots.reshape(-1),
        cache.step,
        n,
        BLOCK=triton.next_power_of_2(max(n, 1)),
    )


@triton.jit
def _protect_slots_kernel(usage_ptr, slots_ptr, step_ptr, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    lane = off < n
    s = tl.load(slots_ptr + off, mask=lane, other=-1)
    step = tl.load(step_ptr) + 1
    tl.store(usage_ptr + s, step, mask=lane & (s >= 0))


def ensure_experts_hybrid(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, fetch_fraction: float = 0.0
) -> None:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded to
    the integer that makes the PCIe fetch and the CPU overflow compute finish closest to
    together. ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats)."""
    # Q16 fixed point so the GPU kernel and the CPU reference cap identically (no float).
    frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
    if not expert_ids.is_cuda:
        return _ensure_experts_hybrid_cpu(cache, layer_id, expert_ids, max_fetch, frac_q16)
    _ensure_experts_hybrid_gpu(cache, layer_id, expert_ids, max_fetch, frac_q16)


def prefill_hit_compact(cache, layer_id: int, buffer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= 2 * num_experts (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / cache slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        buffer_id * num_experts,
        2 * num_experts,
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def materialize_layer(cache, layer_id: int) -> None:
    _materialize_layer_gpu(cache, layer_id)


def reset_cache(cache) -> None:
    _reset_cache_gpu(cache)






def _ensure_experts_hybrid_gpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int
) -> None:
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    num_warps = 8 if block_c >= 2048 else 4
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        num_warps=num_warps,
    )


def _ensure_experts_hybrid_cpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int
) -> None:
    """CPU reference mirror of the hybrid kernel (eviction/fetch decisions bit-identical to
    the GPU path; see tests/test_offload_lru_kernels.py). Fetches at most ``max_fetch`` (or
    the bandwidth-matched ``~frac_q16/2^16 * misses`` when ``frac_q16`` > 0) of the missing
    experts; overflow misses are rewritten to -1. With ``BY_RECENCY`` the fetch set is the
    highest-current-route-count, most-recently-active misses (ties -> lower id);
    else the lowest ids."""
    flat_experts = expert_ids.view(-1).tolist()
    seen = []
    for expert in flat_experts:
        if expert not in seen:
            seen.append(expert)

    cache.active_mask.zero_()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)
    for expert in seen:
        cache.active_mask[expert] = 1

    for expert in seen:
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot != -1:
            cache.usage[slot] = step

    missing = [e for e in seen if int(cache.slot_for_id[layer_id, e].item()) == -1]
    route_counts = {e: flat_experts.count(e) for e in seen}
    if _HYBRID_FETCH_BY_RECENCY:
        rec = cache.expert_recency[layer_id].tolist()
        missing.sort(key=lambda e: (-route_counts[e], -rec[e], e))
    else:
        missing.sort()
    if frac_q16 > 0:
        m, q = len(missing), 1 << 16
        lo = (m * frac_q16) >> 16
        cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)  # noqa: E731
        max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
    num_fetch = min(len(missing), int(max_fetch))
    cache.num_missing_full.fill_(len(missing))
    cache.num_indices.fill_(num_fetch)

    usage = cache.usage.tolist()
    for idx in range(num_fetch):
        expert = missing[idx]
        victim = min(range(cache.cache_size), key=lambda s: (usage[s], s))
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        cache.evict_slots[idx] = victim
        cache.src_indices[idx] = expert  # layer-local row

    if _HYBRID_FETCH_BY_RECENCY:
        for expert in seen:
            cache.expert_recency[layer_id, expert] = step

    # Overflow misses keep slot_for_id == -1, so the rewrite below yields -1 for them.
    flat = expert_ids.view(-1)
    for i in range(flat.numel()):
        flat[i] = int(cache.slot_for_id[layer_id, int(flat[i].item())].item())


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BY_RECENCY: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) first prioritizes
    the experts used by the most routes in the current batch, so one PCIe copy removes
    as many repeated CPU GEMVs as possible during a speculative verify. It then uses
    most-recent activity (LRU on the expert, via ``expert_recency``), breaking final ties
    toward the lower expert id. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    route_count = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        route_count += (off_e == e).to(tl.int32)
    is_active = route_count > 0
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (num_missing * fetch_frac_q16) >> 16
        cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
        cost_hi = tl.maximum(
            (lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16
        )
        max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    num_fetch = tl.minimum(num_missing, max_fetch)
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    is_hit = is_active & (slot >= 0)
    tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (current route count desc, recency desc, id asc)
    # into one strictly-ordered score. Previous recency is <= step-1, so multiplying
    # route_count by step+1 makes one additional current route dominate the full recency
    # range. The id term spans only [0, num_experts), breaking exact ties only.
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_missing,
            (route_count.to(tl.int64) * (step + 1) + rec) * num_experts
            + (num_experts - 1 - off_e),
            -1152921504606846976,
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    if num_fetch > 0:
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        oid = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
        u = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
        owner_active = c_mask & False
        for i in tl.range(num_active):
            ei = tl.load(expert_ids_ptr + i)
            owner_active = owner_active | (oid == base + ei)
        u = tl.where(owner_active | (~c_mask), 9223372036854775807, u)
        for i in tl.range(num_fetch):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            old_id = tl.sum(tl.where(off_c == victim, oid, 0))
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, e)  # layer-local row
            u = tl.where(off_c == victim, 9223372036854775807, u)

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        s = tl.load(slot_for_id_ptr + base + e)
        tl.store(expert_ids_ptr + i, s)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(do_not_specialize=["buffer_base"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows, compacted
    src_ptr,      # [num_experts] int32 out: cache slots, compacted
    num_ptr,      # [1] int64 out: hit count
    buffer_base,  # buffer_id * num_experts
    threshold,    # 2 * num_experts
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, (buffer_base + offs).to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))


# ---------------------------------------------------------------- DMA copy-engine path
# Doorbell protocol for --moe-copy-engine (see offload_cache.DmaCopyService): the graph
# increments a device epoch and mirrors the miss list to pinned host memory; a host
# service thread issues copy-engine DMAs and acks by writing the epoch to a device flag
# via the same copy stream (stream order => rows land before the ack); a spin kernel
# holds the compute stream until the ack. All shapes fixed -> CUDA-graph capturable.
#
# Both kernels below are plain CUDA C++ via load_jit, deliberately NOT triton (see
# kernel/csrc/jit/dma_spin.cuh for the full rationale and the kernels themselves).
# Summary: the spin kernel must re-read device memory on every iteration to observe a
# write from a *different host thread's* CUDA stream, which a C++ `volatile` pointer
# dereference guarantees and triton's `tl.load(ptr, volatile=True)` only conventionally
# provides; and -- found while reproducing the actual --moe-copy-engine boot hang (see
# /root/test_stage_wait_debug.py) -- triton kernel LAUNCHES are not safe in this hot
# path at all: with the spin kernel already converted to C++, the doorbell protocol
# still deadlocked in plain eager mode (no CUDA graph capture) once enough
# back-to-back, unsynchronized doorbell rounds queued up that the DmaCopyService daemon
# thread had real copy backlog. A watchdog thread-stack dump caught the MAIN thread
# stuck inside triton's `compiler.py:_init_handles` (via `launch_metadata`), invoked
# from the epoch-bump kernel's launch -- triton's lazy per-launch handle/module
# (re)initialization does not tolerate this concurrent, backlogged, multi-threaded CUDA
# usage pattern on this platform (RTX 5090 / WSL2 GPU-PV / WDDM). Swapping that one
# launch for a non-triton op made the identical repro pass cleanly, so both kernels
# route through load_jit/tvm-ffi instead (the same launch path already used elsewhere
# in this codebase for other hot, repeatedly launched kernels).
@lru_cache(maxsize=None)
def _jit_dma_doorbell_module() -> "Module":
    from freetoken.kernel.utils import load_jit

    return load_jit(
        "dma_doorbell",
        cuda_files=["dma_spin.cuh"],
        cuda_wrappers=[
            ("dma_epoch_bump_cpp", "dma_epoch_bump_cpp"),
            ("dma_spin_wait_cpp", "dma_spin_wait_cpp"),
        ],
    )


def dma_epoch_bump(epoch: torch.Tensor, layer_out: torch.Tensor, layer_id: int) -> None:
    _jit_dma_doorbell_module().dma_epoch_bump_cpp(epoch, layer_out, layer_id)


def dma_spin_wait(done_addr: int, epoch: torch.Tensor) -> None:
    """``done_addr``: the GPU-visible mapped address of the ack flag (a PINNED HOST
    int64), from ``freetoken.kernel.pinned.device_ptr`` -- NOT a device tensor. See
    kernel/csrc/jit/dma_spin.cuh's file comment for why the ack must be zero-copy
    rather than a normal device buffer updated via cudaMemcpyAsync."""
    _jit_dma_doorbell_module().dma_spin_wait_cpp(int(done_addr), epoch)


# ------------------------------------------------------------- DMA copy-engine daemon
# Round 2: the host side of the doorbell (previously a Python thread, see
# offload_cache.DmaCopyService._serve) is ALSO not safe to leave in Python. Real-
# hardware replay of a captured graph carrying many doorbell rounds deadlocked even
# after round 1's fixes: `cudaGraphLaunch` for a multi-round graph can itself block
# inside the driver under the same WDDM/GPU-PV submission-queue pressure, and -- unlike
# a documented blocking call such as cudaStreamSynchronize -- a normally-async graph
# launch has no established reason for PyTorch's binding to release the GIL around it.
# If it doesn't, the Python daemon thread can never run again (not even to read a
# pinned int), so it never sees the doorbell and never acks: spin waits on the ack: the
# ack needs Python; Python needs the GIL; the GIL is held by the blocked launch; the
# launch is blocked behind the spin. A plain std::thread never touches the GIL at all,
# so it can't be starved by it -- see kernel/csrc/jit/dma_service.cuh for the daemon
# loop itself (same protocol as the old Python one: poll the pinned epoch, issue
# cudaMemcpyAsync per (bank, row) on its own stream). The GIL fix alone was not
# sufficient, though: see dma_spin.cuh's file comment for why the ack itself had to
# stop being a CUDA API call (a bare pinned-host store instead) to actually resolve
# the real-hardware replay hang.
@lru_cache(maxsize=None)
def _jit_dma_service_module() -> "Module":
    from freetoken.kernel.utils import load_jit

    return load_jit(
        "dma_service",
        cuda_files=["dma_service.cuh"],
        cuda_wrappers=[
            ("dma_service_start", "dma_service_start"),
            ("dma_service_stop", "dma_service_stop"),
        ],
    )


def dma_service_start(
    host_ptrs: torch.Tensor,
    row_bytes: torch.Tensor,
    slot_ptrs: torch.Tensor,
    h_epoch: torch.Tensor,
    h_layer: torch.Tensor,
    h_num: torch.Tensor,
    h_slots: torch.Tensor,
    h_rows: torch.Tensor,
    h_ack: torch.Tensor,
    h_error: torch.Tensor,
    num_layers: int,
    num_banks: int,
    max_rows: int,
    device_id: int,
) -> int:
    """Start the C++ doorbell daemon; returns an opaque handle for dma_service_stop.

    ``host_ptrs`` is a CPU int64 tensor [num_banks, num_layers] of each bank/layer's
    host source row-array base address (``per_layer[L].data_ptr()``); ``row_bytes`` a
    CPU int64 [num_banks] of each bank's per-row byte size; ``slot_ptrs`` a CPU int64
    [num_banks] of each bank's device slot-cache base address. The pinned h_* tensors
    are the same mirrors/doorbell machinery DmaCopyService.stage_and_wait writes into;
    the daemon acks by writing directly into ``h_ack`` (a bare host store -- see
    dma_spin.cuh, which reads it back through its GPU-visible mapped alias, not a
    separate device buffer). ``h_error`` is a zero-initialized pinned int64 flag the
    daemon sets (non-zero) instead of raising in a thread nothing is watching for
    exceptions.
    """
    return int(
        _jit_dma_service_module().dma_service_start(
            host_ptrs, row_bytes, slot_ptrs, h_epoch, h_layer, h_num, h_slots, h_rows,
            h_ack, h_error, int(num_layers), int(num_banks), int(max_rows),
            int(device_id),
        )
    )


def dma_service_stop(handle: int) -> None:
    _jit_dma_service_module().dma_service_stop(int(handle))
