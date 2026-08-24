"""Decode-time MoE routing trace -- research instrumentation, env-gated.

Enabled ONLY when ``FREETOKEN_ROUTING_TRACE`` names a path prefix at process
start (``FREETOKEN_ROUTING_TRACE=/tmp/run1`` writes ``/tmp/run1.bin``).  With the
variable unset ``get_tracer()`` returns ``None`` at import time, every hook site
short-circuits on a stored ``is None`` check, and the captured CUDA graph is the
untraced one byte for byte: no extra tensor is allocated, no extra kernel is
enqueued, no other module state changes.

What it answers
---------------
1. *Prefetch feasibility.*  At every MoE layer ``L`` we additionally run the
   routers of layers ``L+1`` and ``L+2`` on the SAME hidden tensor that feeds
   layer ``L``'s own router, using the model's own ``Gate`` modules (their
   weights are small, GPU-resident bf16 tensors), so the scoring function is
   reproduced exactly.  Comparing those predictions against the *actual* top-k
   recorded one/two layers later measures how far ahead an expert prefetcher
   could see.
2. *Offline cache-policy simulation.*  The full per-step, per-layer (expert id,
   gate weight) sequence is dumped so LRU / Belady / prefetch policies can be
   replayed offline.

Graph safety
------------
Everything recorded inside the captured region is a fixed-shape device copy into
a persistent buffer that is allocated ONCE, before capture, at a fixed address
(the engine runs an eager decode warm-up forward immediately before
``torch.cuda.graph(...)``; ``_alloc`` asserts it is not itself running under
capture).  There is no data-dependent host branching: which lookahead gates
exist is decided by ``layer_id`` alone, which is a capture-time constant.  The
D2H drain happens on the host AFTER ``graph.replay()`` returns
(``GraphRunner.replay``), into a pinned ring buffer, and a writer thread appends
to the file once the copy's event has completed.

Only decode steps are recorded; prefill and speculative-verify batches ride the
prefill phase and are skipped by the caller.  Only batch position 0 is recorded
(the research traffic is single-request).

Binary format (little endian)
-----------------------------
File header, 32 bytes, written once::

    magic         char[8]  b"FTROUTE1"
    version       uint32   = 1
    n_layers      uint32   number of MoE layers traced (L)
    top_k         uint32   experts selected per layer (K)
    n_experts     uint32   routed experts per layer (for id-range sanity)
    record_bytes  uint32   = 8 + L * 4 * K * 2
    reserved      uint32   = 0

Then ``record_bytes``-sized records, one per traced decode step::

    step          uint32   monotonic decode-step counter; GAPS MEAN DROPPED
                           RECORDS (ring exhaustion), never reordering
    batch_size    uint16   decode batch size of the step (only row 0 is traced)
    flags         uint16   = 0
    payload       int16[L][4][K]
                     [L][0] actual top-k expert ids at layer L
                     [L][1] top-k ids predicted by layer L+1's router run on
                            layer L's router input  (-1 when L+1 does not exist)
                     [L][2] top-k ids predicted by layer L+2's router run on
                            layer L's router input  (-1 when L+2 does not exist)
                     [L][3] actual top-k gate weights, float16 BIT PATTERN
                            (reinterpret the int16 slots as float16)

For DeepSeek-V4-Flash (L=43, K=6) a record is 8 + 2064 = 2072 bytes, so the
per-step D2H is 2064 bytes -- far below the 100KB budget.

See ``/root/trace_reader.py`` for the parser / analysis entry point.
"""

from __future__ import annotations

import atexit
import os
import queue
import struct
import threading

import torch

__all__ = [
    "FORMAT_VERSION",
    "HEADER",
    "MAGIC",
    "RECORD_HEADER",
    "RoutingTracer",
    "get_tracer",
]

MAGIC = b"FTROUTE1"
FORMAT_VERSION = 1

# magic, version, n_layers, top_k, n_experts, record_bytes, reserved
HEADER = struct.Struct("<8s6I")
# step, batch_size, flags
RECORD_HEADER = struct.Struct("<IHH")

# Payload slots per layer (see the module docstring's format section).
SLOT_ACTUAL_IDS = 0
SLOT_PRED_L1 = 1
SLOT_PRED_L2 = 2
SLOT_ACTUAL_W = 3
N_SLOTS = 4

# Pinned staging buffers. One is consumed per decode step and returned by the
# writer thread once its copy event has completed; 256 x ~2KB is 0.5MB of pinned
# host memory and hundreds of steps of slack over a writer that only ever does a
# 2KB append.
RING_SLOTS = 256

# Environment variable holding the output path PREFIX (the trace lands at
# ``<prefix>.bin``). Read exactly once, at import.
ENV_VAR = "FREETOKEN_ROUTING_TRACE"


class RoutingTracer:
    """Owns the persistent device record buffer, the pinned drain ring and the
    writer thread. One instance per process; see ``get_tracer()``."""

    def __init__(self, prefix: str, ring_slots: int = RING_SLOTS) -> None:
        self.prefix = prefix
        self.path = prefix + ".bin"
        self.ring_slots = ring_slots

        # Geometry, learned from the first registered layer and asserted after.
        self.n_layers = 0
        self.top_k = 0
        self.n_experts = 0
        self._gates: dict[int, torch.nn.Module] = {}

        # Persistent device buffer (fixed address, captured into the graph).
        self._dev_i16: torch.Tensor | None = None
        self._dev_f16: torch.Tensor | None = None
        self._dev_flat: torch.Tensor | None = None

        # Host drain.
        self._host: list[torch.Tensor] = []
        self._events: list[torch.cuda.Event] = []
        self._free: queue.Queue[int] = queue.Queue()
        self._work: queue.Queue[tuple | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._fh = None

        self.step = 0
        self.written = 0
        self.dropped = 0
        # Records are only drained once ``arm()`` has run, i.e. after CUDA-graph
        # capture: the capture path runs eager warm-up decode forwards on a dummy
        # request whose routing is meaningless and must not pollute the trace.
        self.armed = False
        self._closed = False
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Model-side wiring (construction time, before any forward)
    # ------------------------------------------------------------------

    def register_gate(
        self,
        layer_id: int,
        gate: torch.nn.Module,
        *,
        n_layers: int,
        top_k: int,
        n_experts: int,
    ) -> None:
        """Publish layer ``layer_id``'s router so layers ``L-1``/``L-2`` can run it.

        Layer ids at or beyond ``n_layers`` (a speculative drafter continues the
        target's layer id space) are ignored: they are neither traced nor used as
        a lookahead target, so the trace stays a clean picture of the target stack.
        """
        if layer_id >= n_layers:
            return
        if self.n_layers == 0:
            self.n_layers, self.top_k, self.n_experts = n_layers, top_k, n_experts
        else:
            assert (self.n_layers, self.top_k, self.n_experts) == (
                n_layers,
                top_k,
                n_experts,
            ), "routing trace does not support mixed MoE geometries in one process"
        self._gates[layer_id] = gate

    def traces_layer(self, layer_id: int) -> bool:
        return layer_id in self._gates

    # ------------------------------------------------------------------
    # Captured region
    # ------------------------------------------------------------------

    def _alloc(self, device: torch.device) -> None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "routing trace buffer would be allocated inside a CUDA graph's private "
            "mempool; it must be allocated by the eager warm-up decode forward that "
            "precedes capture"
        )
        assert self.n_layers > 0, "no MoE gate registered with the routing tracer"
        # -1 marks "never written" (the missing L+1/L+2 predictions of the last
        # two layers): the buffer is persistent, so those slots keep the sentinel
        # for the lifetime of the process.
        self._dev_i16 = torch.full(
            (self.n_layers, N_SLOTS * self.top_k), -1, dtype=torch.int16, device=device
        )
        # Same storage seen as float16 -- the weight slots hold fp16 bit patterns.
        self._dev_f16 = self._dev_i16.view(torch.float16)
        self._dev_flat = self._dev_i16.view(-1)

    def record(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Write layer ``layer_id``'s record and its two lookahead predictions.

        ``hidden`` is the exact ``[T, dim]`` tensor this layer's own router
        consumed; ``input_ids`` the matching flat token ids (the first
        ``n_hash_layers`` routers are hash routers keyed on the token, not the
        hidden state). ``weights``/``indices`` are the router's own outputs.
        All device work here is fixed-shape, so it captures cleanly.
        """
        if self._dev_i16 is None:
            self._alloc(hidden.device)
        assert self._dev_i16 is not None and self._dev_f16 is not None
        k = self.top_k
        base = layer_id
        self._dev_i16[base, SLOT_ACTUAL_IDS * k : (SLOT_ACTUAL_IDS + 1) * k].copy_(
            indices[0, :k]
        )
        self._dev_f16[base, SLOT_ACTUAL_W * k : (SLOT_ACTUAL_W + 1) * k].copy_(
            weights[0, :k]
        )
        # Which lookahead gates exist is a function of layer_id alone -- a
        # capture-time constant, not a device value. Nothing here branches on data.
        for ahead, slot in ((1, SLOT_PRED_L1), (2, SLOT_PRED_L2)):
            gate = self._gates.get(layer_id + ahead)
            if gate is None:
                continue
            _, pred = gate(hidden, input_ids)
            self._dev_i16[base, slot * k : (slot + 1) * k].copy_(pred[0, :k])

    # ------------------------------------------------------------------
    # Host drain (post-replay)
    # ------------------------------------------------------------------

    def arm(self) -> None:
        """Open the file and start draining. Called once, after graph capture."""
        if self.armed or self._closed:
            return
        if self.n_layers == 0:
            return  # no MoE model in this process; stay inert
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._fh.write(
            HEADER.pack(
                MAGIC,
                FORMAT_VERSION,
                self.n_layers,
                self.top_k,
                self.n_experts,
                self.record_bytes,
                0,
            )
        )
        self._fh.flush()
        width = self.n_layers * N_SLOTS * self.top_k
        for i in range(self.ring_slots):
            self._host.append(
                torch.empty(width, dtype=torch.int16, pin_memory=True)
            )
            # blocking=True: the writer thread waits in the driver instead of
            # spinning a core against the decode loop.
            self._events.append(torch.cuda.Event(blocking=True))
            self._free.put(i)
        self._thread = threading.Thread(
            target=self._writer_loop, name="routing-trace-writer", daemon=True
        )
        self._thread.start()
        self.armed = True

    @property
    def record_bytes(self) -> int:
        return RECORD_HEADER.size + self.n_layers * N_SLOTS * self.top_k * 2

    def after_step(self, batch_size: int = 1) -> None:
        """Queue the just-replayed step's records for the writer thread.

        Called on the engine stream right after ``graph.replay()`` returns, so the
        D2H is stream-ordered behind the graph's own writes. The copy is async;
        an event marks its completion and the writer thread waits on that, which
        keeps the decode loop free of any synchronization.
        """
        if not self.armed or self._dev_i16 is None:
            return
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            # Writer fell behind. Drop rather than stall decode; the missing step
            # number makes the gap visible to the reader.
            self.dropped += 1
            self.step += 1
            return
        assert self._dev_flat is not None
        self._host[slot].copy_(self._dev_flat, non_blocking=True)
        event = self._events[slot]
        event.record()
        self._work.put((slot, event, self.step, int(batch_size)))
        self.step += 1

    def _writer_loop(self) -> None:
        assert self._fh is not None
        while True:
            item = self._work.get()
            if item is None:
                return
            slot, event, step, batch_size = item
            event.synchronize()
            self._fh.write(RECORD_HEADER.pack(step, batch_size & 0xFFFF, 0))
            self._fh.write(self._host[slot].numpy().tobytes())
            self._fh.flush()
            self.written += 1
            self._free.put(slot)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.armed = False
        if self._thread is not None:
            self._work.put(None)
            self._thread.join(timeout=10.0)
            self._thread = None
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


_TRACER: RoutingTracer | None = None
_PREFIX = os.environ.get(ENV_VAR, "").strip()
if _PREFIX:
    _TRACER = RoutingTracer(_PREFIX)


def get_tracer() -> RoutingTracer | None:
    """The process-wide tracer, or ``None`` when ``FREETOKEN_ROUTING_TRACE`` was
    unset at import. Callers cache the result and guard every hook on it."""
    return _TRACER
