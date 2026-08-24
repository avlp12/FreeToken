"""Host-RAM (and optional disk) KV tier for the DSV4 swa_radix path (session swap).

When the SWA radix tree evicts a node's FULL KV (the moment its tokens would need a cold
re-prefill to ever be reused), the eviction hook snapshots the node's page-aligned span into
host RAM instead of just dropping it: the compressed-KV and indexer rows (arithmetic
``full_loc // ratio`` addressing, so they relocate freely), and -- when the span's window is
still live -- the window rows plus both compress-state carry rings. On a later admission whose
prompt extends past the GPU tree's frontier, ``CacheManager`` restores matching spans into
freshly allocated pages and re-attaches them to the tree, so the request resumes from
``cached_len`` instead of re-prefilling the whole session.

With a disk budget (``--host-kv-disk-gb``), entries are additionally flushed to
``~/.cache/freetoken/hostkv/<model>/`` by a background writer, RAM-trimmed entries keep a
disk-backed stub, and the index is reloaded at startup -- sessions survive a server restart.
The scheduler's idle-time ``flush_tree_to_host`` snapshots not-yet-saved live tree spans so a
restart does not lose the sessions that were never evicted.

Content is stored relocatable (no absolute slot ids), keyed by the token-id prefix. Spans
restored without a live window re-enter the tree as ``swa_tombstone`` nodes -- exactly the
state they were evicted in. The tier is strictly best-effort: every save/restore is wrapped by
the caller so a failure degrades to the old evict-and-recompute behavior, never a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_PAYLOAD_KEYS = ("cmp", "idx", "win", "attn_ring", "idx_ring")


@dataclass
class HostKVEntry:
    ids: torch.Tensor                     # CPU prefix token ids [0:end] (always resident)
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
    in_ram: bool = True                   # payload resident (False = disk-backed stub)
    on_disk: bool = False                 # .meta/.data files exist and are current
    path: str = ""                        # data-file basename (no extension)

    def span(self) -> int:
        return self.end - self.start


class HostKVTier:
    """Best-effort host store of evicted DSV4 KV spans: RAM working set (LRU-bounded by
    ``budget_bytes``) over an optional disk tier (LRU-bounded by ``disk_budget_bytes``)."""

    def __init__(self, pool, page_size: int, budget_bytes: int,
                 disk_dir: str | None = None, disk_budget_bytes: int = 0,
                 fingerprint: str = "") -> None:
        self.pool = pool                  # DSV4PagedKVCache (duck-typed: cmp_pool/idx_pool/...)
        self.page_size = int(page_size)
        self.budget = int(budget_bytes)
        self.disk_budget = int(disk_budget_bytes)
        self.fingerprint = fingerprint
        self.entries: List[HostKVEntry] = []
        self.total_ram_bytes = 0
        self.total_disk_bytes = 0
        self._clk = 0
        self.saved_tokens = 0             # observability
        self.restored_tokens = 0
        self.restore_hits = 0
        self.disk_loads = 0
        self._lock = threading.Lock()
        self.disk_dir = None
        self._flushq: "queue.Queue[HostKVEntry]" = queue.Queue()
        if disk_dir is not None:
            self.disk_dir = disk_dir
            os.makedirs(disk_dir, exist_ok=True)
            if self.disk_budget > 0:
                self._load_disk_index()
                threading.Thread(
                    target=self._writer_loop, name="hostkv-writer", daemon=True
                ).start()
        self._write_stats()

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
        with self._lock:
            self._clk += 1
            entry.stamp = self._clk
            # Replace an identical-span older entry (re-eviction of a restored span).
            for old in [e for e in self.entries
                        if e.start == entry.start and e.end == entry.end
                        and torch.equal(e.ids, entry.ids)]:
                self._drop_entry(old, drop_disk=False)
                entry.path, entry.on_disk = old.path, False  # rewrite the file
            self.entries.append(entry)
            self.total_ram_bytes += entry.nbytes
            self.saved_tokens += span
            if self.disk_dir is not None and self.disk_budget > 0:
                self._flushq.put(entry)
            self._trim_ram()
        self._write_stats()
        logger.info(
            f"host-kv: saved span [{entry.start}:{entry.end}) live={entry.live} "
            f"({entry.nbytes >> 20} MiB, ram {self.total_ram_bytes >> 20} MiB, "
            f"{len(self.entries)} entries)"
        )

    def has(self, ids_cpu: torch.Tensor, start: int, end: int) -> bool:
        with self._lock:
            return any(
                e.start == start and e.end == end and torch.equal(e.ids, ids_cpu[:end])
                for e in self.entries
            )

    # ------------------------------------------------------------------ find / restore
    def find(self, input_ids: torch.Tensor, frontier: int) -> Optional[HostKVEntry]:
        """The stored span starting exactly at ``frontier`` whose ids match ``input_ids``;
        longest span wins. ``input_ids`` and stored ids are CPU tensors."""
        best = None
        n = len(input_ids)
        with self._lock:
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
        if not entry.in_ram:
            self._load_payload(entry)
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
        self.restore_hits += 1
        self._write_stats()
        return live

    # ------------------------------------------------------------------ disk tier
    def _entry_name(self, entry: HostKVEntry) -> str:
        h = hashlib.sha1(entry.ids.numpy().tobytes()).hexdigest()[:24]
        return f"{h}_{entry.start}_{entry.end}"

    def _writer_loop(self) -> None:
        while True:
            entry = self._flushq.get()
            try:
                self._flush_entry(entry)
            except Exception as exc:
                logger.warning(f"host-kv: disk flush failed: {exc!r}")

    def _flush_entry(self, entry: HostKVEntry) -> None:
        if not entry.in_ram:
            return
        name = entry.path or self._entry_name(entry)
        meta = {
            "ids": entry.ids, "start": entry.start, "end": entry.end,
            "live": entry.live, "nbytes": entry.nbytes, "fingerprint": self.fingerprint,
        }
        data = {k: getattr(entry, k) for k in _PAYLOAD_KEYS}
        tmp = os.path.join(self.disk_dir, name + ".tmp")
        torch.save(data, tmp)
        os.replace(tmp, os.path.join(self.disk_dir, name + ".data"))
        torch.save(meta, tmp)
        os.replace(tmp, os.path.join(self.disk_dir, name + ".meta"))
        with self._lock:
            entry.path, entry.on_disk = name, True
            self.total_disk_bytes = self._disk_usage()
            self._trim_disk()
        self._write_stats()
        logger.info(f"host-kv: flushed span [{entry.start}:{entry.end}) to disk "
                    f"({self.total_disk_bytes >> 20} MiB on disk)")

    def _load_payload(self, entry: HostKVEntry) -> None:
        data = torch.load(
            os.path.join(self.disk_dir, entry.path + ".data"), map_location="cpu",
            weights_only=False,
        )
        for k in _PAYLOAD_KEYS:
            setattr(entry, k, data[k])
        with self._lock:
            entry.in_ram = True
            self.total_ram_bytes += entry.nbytes
            self._trim_ram()
        self.disk_loads += 1

    def _load_disk_index(self) -> None:
        loaded = 0
        for fn in sorted(os.listdir(self.disk_dir)):
            if not fn.endswith(".meta"):
                continue
            name = fn[: -len(".meta")]
            try:
                meta = torch.load(
                    os.path.join(self.disk_dir, fn), map_location="cpu", weights_only=False)
                if meta.get("fingerprint") != self.fingerprint:
                    logger.warning(f"host-kv: skipping {fn} (model fingerprint mismatch)")
                    continue
                if not os.path.exists(os.path.join(self.disk_dir, name + ".data")):
                    continue
                self.entries.append(HostKVEntry(
                    ids=meta["ids"], start=int(meta["start"]), end=int(meta["end"]),
                    live=bool(meta["live"]), nbytes=int(meta["nbytes"]),
                    in_ram=False, on_disk=True, path=name,
                ))
                loaded += 1
            except Exception as exc:
                logger.warning(f"host-kv: skipping unreadable {fn}: {exc!r}")
        self.total_disk_bytes = self._disk_usage()
        if loaded:
            logger.info(
                f"host-kv: loaded {loaded} spans from disk "
                f"({self.total_disk_bytes >> 20} MiB) -- sessions survive the restart")

    def _disk_usage(self) -> int:
        n = 0
        for fn in os.listdir(self.disk_dir):
            if fn.endswith((".data", ".meta")):
                n += os.path.getsize(os.path.join(self.disk_dir, fn))
        return n

    # ------------------------------------------------------------------ trims (lock held)
    def _drop_entry(self, e: HostKVEntry, *, drop_disk: bool) -> None:
        if e in self.entries:
            self.entries.remove(e)
        if e.in_ram:
            self.total_ram_bytes -= e.nbytes
        if drop_disk and e.on_disk:
            for ext in (".data", ".meta"):
                try:
                    os.remove(os.path.join(self.disk_dir, e.path + ext))
                except OSError:
                    pass

    def _trim_ram(self) -> None:
        # Prefer dropping payloads that are safe on disk; entries pending flush stay resident.
        while self.total_ram_bytes > self.budget:
            resident = [e for e in self.entries if e.in_ram]
            backed = [e for e in resident if e.on_disk]
            if backed:
                victim = min(backed, key=lambda e: e.stamp)
                for k in _PAYLOAD_KEYS:
                    setattr(victim, k, {})
                victim.in_ram = False
                self.total_ram_bytes -= victim.nbytes
                continue
            unflushed = [e for e in resident if not e.on_disk]
            if self.disk_budget > 0 and self._flushq.qsize() > 0:
                break  # writer will back these soon; tolerate transient overshoot
            if len(unflushed) <= 1:
                break
            self._drop_entry(min(unflushed, key=lambda e: e.stamp), drop_disk=False)

    def _trim_disk(self) -> None:
        while self.total_disk_bytes > self.disk_budget:
            backed = [e for e in self.entries if e.on_disk]
            if not backed:
                break
            victim = min(backed, key=lambda e: e.stamp)
            drop_whole = not victim.in_ram
            if drop_whole:
                self._drop_entry(victim, drop_disk=True)
            else:
                for ext in (".data", ".meta"):
                    try:
                        os.remove(os.path.join(self.disk_dir, victim.path + ext))
                    except OSError:
                        pass
                victim.on_disk = False
            self.total_disk_bytes = self._disk_usage()

    # ------------------------------------------------------------------ misc
    def _entry_bytes(self, e: HostKVEntry) -> int:
        n = e.ids.numel() * e.ids.element_size()
        for k in _PAYLOAD_KEYS:
            n += sum(t.numel() * t.element_size() for t in getattr(e, k).values())
        return int(n)

    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "ram_bytes": self.total_ram_bytes,
            "disk_bytes": self.total_disk_bytes,
            "saved_tokens": self.saved_tokens,
            "restored_tokens": self.restored_tokens,
            "restore_hits": self.restore_hits,
            "disk_loads": self.disk_loads,
            "updated": time.time(),
        }

    def _write_stats(self) -> None:
        if self.disk_dir is None:
            return
        try:
            tmp = os.path.join(self.disk_dir, "stats.tmp")
            with open(tmp, "w") as f:
                json.dump(self.stats(), f)
            os.replace(tmp, os.path.join(self.disk_dir, "stats.json"))
        except OSError:
            pass


def host_kv_dir(served_model_name: str | None) -> str:
    """The per-model host-KV directory (data files + stats.json), shared by the scheduler
    (writer) and the API server (stats reader)."""
    name = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in (served_model_name or "model"))
    return os.path.join(os.path.expanduser("~/.cache/freetoken/hostkv"), name)


__all__ = ["HostKVTier", "HostKVEntry", "host_kv_dir"]
