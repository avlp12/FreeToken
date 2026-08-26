"""Env-gated Programmatic Dependent Launch (PDL) for FreeToken's Triton decode kernels.

PDL (`cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization`, sm_90+)
lets a kernel's grid be *launched* while its stream predecessor is still draining, so the
CTA scheduling, parameter loads and index math happen off the critical path. Two device
markers bracket the part that genuinely depends on the predecessor:

    gdc_wait()                -> `griddepcontrol.wait`, placed immediately before the first
                                 read of predecessor-produced data (NOT at the top of the
                                 kernel -- the point is to do address math before it).
    gdc_launch_dependents()   -> `griddepcontrol.launch_dependents`, placed after the last
                                 write a successor needs.

Measured on this box (RTX 5090, sm_120, WSL2, CUDA 13.3) with a captured-graph chain: the
per-node dispatch floor drops from ~1.04 us to ~0.66 us, i.e. ~0.30-0.39 us saved per
enrolled node. The saving scales with the number of *enrolled* nodes and does not require
the neighbours to be enrolled -- a PDL kernel whose predecessor is a plain vendored ATen
kernel still gets ~0.27-0.35 us, because the win is the dependent grid's own launch being
overlapped, not the predecessor releasing early.

Gate
----
`FREETOKEN_PDL=1` turns the *new* enrollments on; default OFF so production is unaffected
until the A/B has been graded. Note this gate deliberately does NOT cover the pre-existing
PDL in `kernel/triton/norm.py`, `activation.py` and `sampling.py`: those shipped and were
graded already, and re-gating them would change the measured baseline.

The value is cached, so it is constant for the process lifetime -- required, because a
CUDA graph captures whichever launch path was taken at capture time and replays it forever.

__restrict__
------------
There is no Triton analogue of the C++ hazard that llama.cpp PR #24030 fixed (`__restrict__`
lets the compiler hoist loads above `griddepcontrol.wait`, turning PDL into a silent race).
Triton emits its own aliasing metadata and `tl.load` is not reordered across the inline
`griddepcontrol.wait`, which carries a memory clobber. The C++ side of this change handles
the restrict question separately -- see `csrc/include/freetoken/utils.cuh`.
"""

from __future__ import annotations

import functools
import os

# Re-exported so an enrolled kernel can do `from ..pdl import gdc_wait,
# gdc_launch_dependents` and have the intrinsics resolve in its own module globals, which
# is where Triton looks them up at JIT-compile time.
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait  # noqa: F401

from freetoken.utils.arch import is_sm90_supported

PDL_ENV = "FREETOKEN_PDL"
_TRUE_VALUES = {"1", "true", "yes", "on"}

# Enrollment tiers. Measured per-kernel on this box (see the note below), PDL is NOT
# uniformly a win, so the gate is graded rather than boolean:
#
#   LIGHT -- small, short, low-CTA kernels (norms, swiglu, gated pool, splitk reduce,
#            act-quant). These win consistently: -0.25 to -0.44 us/node, because their
#            own execution is short enough that the launch latency is the dominant term.
#   GEMV  -- the FP8/BF16 projection GEMVs. These are 2k-CTA, multi-microsecond kernels
#            at the shipped `_FUSED_DECODE_CFG` shapes, and at those shapes PDL measures
#            as a LOSS (+4.8% to +12.6% on 5 of 7 production shapes): the successor's
#            pre-launched CTAs spin on `griddepcontrol.wait` while occupying SM slots the
#            still-resident predecessor needs. Kept behind level 2 so the regression is
#            opt-in and can be A/B'd separately rather than silently bundled.
LEVEL_OFF = 0
LEVEL_LIGHT = 1
LEVEL_GEMV = 2


@functools.cache
def pdl_level() -> int:
    """0 = off, 1 = light kernels only, 2 = light + projection GEMVs.

    Cached on purpose: a CUDA graph bakes in the launch path chosen during capture, so
    this must not be able to change between capture and replay.
    """
    raw = os.getenv(PDL_ENV, "0").strip().lower()
    if not is_sm90_supported():
        return LEVEL_OFF
    if raw in _TRUE_VALUES:
        level = LEVEL_LIGHT
    elif raw.isdigit():
        level = min(int(raw), LEVEL_GEMV)
    else:
        level = LEVEL_OFF
    if level:
        # One line per process, so a server log proves which arm actually ran.
        print(f"[freetoken] PDL enabled: level={level} "
              f"({'light' if level == LEVEL_LIGHT else 'light+gemv'})", flush=True)
    return level


def pdl_enabled(tier: int = LEVEL_LIGHT) -> bool:
    """True iff PDL is on for a call site enrolled at ``tier``."""
    return pdl_level() >= tier


__all__ = [
    "gdc_launch_dependents", "gdc_wait", "pdl_enabled", "pdl_level",
    "PDL_ENV", "LEVEL_OFF", "LEVEL_LIGHT", "LEVEL_GEMV",
]
