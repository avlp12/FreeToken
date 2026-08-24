"""Host-RAM KV tier for the DSV4 swa_radix path (session swap).

When the SWA radix tree evicts a node's FULL KV (the moment its tokens would need a cold
re-prefill to ever be reused), the eviction hook snapshots the node's page-aligned span into
host RAM instead of just dropping it: the compressed-KV and indexer rows (arithmetic
``full_loc // ratio`` addressing, so they relocate freely), and -- when the span's window is
still live -- the window rows plus both compress-state carry rings. On a later admission whose
prompt extends past the GPU tree's frontier, ``CacheManager`` restores matching spans into
freshly allocated pages and re-attaches them to the tree, so the request resumes from
``cached_len`` instead of re-prefilling the whole session.

Content is stored relocatable (no absolute slot ids), keyed by the token-id prefix. Spans
restored without a live window re-enter the tree as ``swa_tombstone`` nodes -- exactly the
state they were evicted in. The tier is strictly best-effort: every save/restore is wrapped by
the caller so a failure degrades to the old evict-and-recompute behavior, never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


@dataclass
class HostKVEntry:
    ids: torch.Tensor                     # CPU prefix token ids [0:end]
    start: int                            # span [start:end), page-aligned
    end: int
    live: bool                            # whole-span window live at save time
    cmp: Dict[int, torch.Tensor] = field(default_factory=dict)   # layer -> [span//r, head]
    idx: Dict[int, torch.Tensor] = field(default_factory=dict)   # layer -> [span//4, ihead]
    win: Dict[int, torch.Tensor] = field(default_factory=dict)   # layer -> [span, head]
    attn_ring: Dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [pages, ring, w]
    idx_ring: Dict[int, torch.Tensor] = field(default_factory=dict)
    nbytes: int = 0
    stamp: int = 0

    def span(self) -> int:
        return self.end - self.start


class HostKVTier:
    """Best-effort host-RAM store of evicted DSV4 KV spans, LRU-bounded by ``budget_bytes``."""

    def __init__(self, pool, page_size: int, budget_bytes: int) -> None:
        self.pool = pool                  # DSV4PagedKVCache (duck-typed: cmp_pool/idx_pool/...)
        self.page_size = int(page_size)
        self.budget = int(budget_bytes)
        self.entries: List[HostKVEntry] = []
        self.total_bytes = 0
        self._clk = 0
        self.saved_tokens = 0             # observability
        self.restored_tokens = 0

    # ------------------------------------------------------------------ save
    def save_span(self, ids_cpu: torch.Tensor, start: int, full_locs: torch.Tensor) -> None:
        """Snapshot the span backed by ``full_locs`` (page-aligned node value) to host RAM.
        ``ids_cpu`` is the FULL prefix [0:end) through the span end. Liveness is read off the
        pool's full->window mapping (still bound at eviction-hook time)."""
        P = self.page_size
        span = int(full_locs.numel())
        if span == 0 or span % P != 0:
            return
        pool = self.pool
        fl = full_locs.to(dtype=torch.int64)
        pages = fl.view(-1, P)
        fbases = pages[:, 0]
        # Defensive: spans must be whole contiguous-ascending pages (page-atomic invariant).
        if not bool((fbases % P == 0).all()):
            return
        if not torch.equal(
            pages, fbases[:, None] + torch.arange(P, device=fbases.device)
        ):
            return
        end = int(start) + span
        entry = HostKVEntry(ids=ids_cpu[:end].clone(), start=int(start), end=end, live=False)

        # Compressed + indexer tiers: pure arithmetic rows, always saved.
        for L, ratio in enumerate(pool.compress_ratios):
            if ratio == 0:
                continue
            rows = (
                fbases[:, None] // ratio
                + torch.arange(P // ratio, device=fbases.device)
            ).flatten()
            entry.cmp[L] = pool.cmp_pool[L].index_select(0, rows).cpu()
            if ratio == 4 and pool.idx_pool[L] is not None:
                entry.idx[L] = pool.idx_pool[L].index_select(0, rows).cpu()

        # Window tier: only when EVERY page of the span is still window-bound. A partially
        # live span re-enters the tree as a tombstone (window content already gone anyway).
        ws = pool.full_to_window[fl]
        if bool((ws.view(-1, P)[:, 0] >= 0).all()):
            entry.live = True
            wbases = ws.view(-1, P)[:, 0]
            for L in range(pool.num_layers):
                entry.win[L] = pool.window_pool[L].index_select(0, ws).cpu()
                ratio = pool.compress_ratios[L]
                if ratio == 0:
                    continue
                ring = pool.state_ring[L]
                base_rows = (wbases // P) * ring.ring_size
                entry.attn_ring[L] = ring.get_blocks(base_rows).cpu()
                if ratio == 4 and pool.indexer_state_ring[L] is not None:
                    iring = pool.indexer_state_ring[L]
                    entry.idx_ring[L] = iring.get_blocks((wbases // P) * iring.ring_size).cpu()

        entry.nbytes = self._entry_bytes(entry)
        self._clk += 1
        entry.stamp = self._clk
        # Replace an identical-span older entry (re-eviction of a restored span).
        self.entries = [
            e for e in self.entries
            if not (e.start == entry.start and e.end == entry.end
                    and torch.equal(e.ids, entry.ids))
        ]
        self.entries.append(entry)
        self.total_bytes = sum(e.nbytes for e in self.entries)
        self.saved_tokens += span
        self._trim()
        logger.info(
            f"host-kv: saved span [{entry.start}:{entry.end}) live={entry.live} "
            f"({entry.nbytes >> 20} MiB, store {self.total_bytes >> 20} MiB, "
            f"{len(self.entries)} entries)"
        )

    # ------------------------------------------------------------------ find / restore
    def find(self, input_ids: torch.Tensor, frontier: int) -> Optional[HostKVEntry]:
        """The stored span starting exactly at ``frontier`` whose ids match ``input_ids``;
        longest span wins. ``input_ids`` and stored ids are CPU tensors."""
        best = None
        n = len(input_ids)
        for e in self.entries:
            if e.start != frontier or e.end > n:
                continue
            if best is not None and e.end <= best.end:
                continue
            if torch.equal(e.ids, input_ids[: e.end]):
                best = e
        if best is not None:
            self._clk += 1
            best.stamp = self._clk
        return best

    def restore_span(self, entry: HostKVEntry, full_locs: torch.Tensor, window_ok: bool) -> bool:
        """Write the entry's content into freshly allocated ``full_locs``. When ``window_ok``
        the caller has already bound window pages for the span (alloc_swa). Returns whether
        the span was restored window-live."""
        P = self.page_size
        pool = self.pool
        fl = full_locs.to(dtype=torch.int64)
        fbases = fl.view(-1, P)[:, 0]
        dev = pool.device

        for L, t in entry.cmp.items():
            ratio = pool.compress_ratios[L]
            rows = (
                fbases[:, None] // ratio + torch.arange(P // ratio, device=fbases.device)
            ).flatten()
            pool.cmp_pool[L].index_copy_(0, rows, t.to(dev, non_blocking=True))
        for L, t in entry.idx.items():
            rows = (fbases[:, None] // 4 + torch.arange(P // 4, device=fbases.device)).flatten()
            pool.idx_pool[L].index_copy_(0, rows, t.to(dev, non_blocking=True))

        live = bool(entry.live and window_ok)
        if live:
            ws = pool.full_to_window[fl]
            assert bool((ws >= 0).all()), "restore_span: window pages not bound by caller"
            wbases = ws.view(-1, P)[:, 0]
            for L, t in entry.win.items():
                pool.window_pool[L].index_copy_(0, ws, t.to(dev, non_blocking=True))
            for L, t in entry.attn_ring.items():
                ring = pool.state_ring[L]
                ring.set_blocks((wbases // P) * ring.ring_size, t.to(dev, non_blocking=True))
            for L, t in entry.idx_ring.items():
                iring = pool.indexer_state_ring[L]
                iring.set_blocks((wbases // P) * iring.ring_size, t.to(dev, non_blocking=True))
        self.restored_tokens += entry.span()
        return live

    # ------------------------------------------------------------------ misc
    def _entry_bytes(self, e: HostKVEntry) -> int:
        n = e.ids.numel() * e.ids.element_size()
        for d in (e.cmp, e.idx, e.win, e.attn_ring, e.idx_ring):
            n += sum(t.numel() * t.element_size() for t in d.values())
        return int(n)

    def _trim(self) -> None:
        while self.total_bytes > self.budget and len(self.entries) > 1:
            victim = min(self.entries, key=lambda e: e.stamp)
            self.entries.remove(victim)
            self.total_bytes -= victim.nbytes
            logger.debug(f"host-kv: dropped LRU span [{victim.start}:{victim.end})")

    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "bytes": self.total_bytes,
            "saved_tokens": self.saved_tokens,
            "restored_tokens": self.restored_tokens,
        }


__all__ = ["HostKVTier", "HostKVEntry"]
