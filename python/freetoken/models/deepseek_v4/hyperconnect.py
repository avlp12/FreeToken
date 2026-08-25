"""The residual-stream state DSV4's hyper-connection sites hand to one another.

DSV4 keeps ``hc_mult`` parallel copies of the residual. Each sublayer is bracketed by
``hc_pre`` (collapse the copies into the sublayer's input) and ``hc_post`` (re-expand
its output back over the copies), and those two ALWAYS meet: every ``hc_post`` in the
model is immediately followed by another ``hc_pre`` -- the next sublayer's, the next
block's, or the output head's. Nothing between them reads the expanded stream.

So the fused stage (``kernel/triton/dsv4/hc_fused.py``) absorbs a pending ``hc_post``
into the *next* site's kernel, and the stream between the two is never materialised on
its own. :class:`HCState` is what travels along that pipeline; :func:`hc_materialize`
is the flush, used only by the DSpark auxiliary taps (which genuinely need a block's
own output) and by the unfused reference path.

Shared by ``model.py`` and ``dspark.py``, which cannot import each other at module
scope.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

from freetoken.kernel.triton.dsv4.hc import hc_post_combine

# Escape hatch: run the hyper-connections as the reference composition
# (hc_post_combine -> inv_rms -> F.linear -> hc_split_sinkhorn -> hc_pre_combine ->
# RMSNorm), one kernel per step, instead of the single fused stage. The numerics tests
# in tests/dsv4/test_hc_fused.py compare the fused stage against exactly this path.
UNFUSED_HC = os.environ.get("FREETOKEN_UNFUSED_HC", "0") not in ("0", "")


class HCState(NamedTuple):
    """The residual stream carried between two hyper-connection sites.

    Exactly one of:

    ``stream``   the flat ``[M, hc_mult*dim]`` residual, already materialised;
    ``pending``  ``(a, stream, post, comb)`` -- the previous sublayer's output ``a`` as
                 ``[M, dim]``, the ``[M, hc_mult*dim]`` residual it branched from, and
                 the re-expand operands the next site will apply.

    ``shape`` is the ``[B, T, hc_mult, dim]`` view the model works in, kept so a flush
    can restore it without threading batch dims through the pipeline.
    """

    shape: torch.Size
    stream: torch.Tensor | None = None
    pending: tuple | None = None

    @classmethod
    def of(cls, x: torch.Tensor) -> "HCState":
        """Wrap a materialised ``[B, T, hc_mult, dim]`` stream."""
        return cls(x.size(), stream=x.reshape(x.shape[0] * x.shape[1], -1))


def hc_materialize(state: HCState) -> torch.Tensor:
    """Flush a deferred re-expand and return the ``[B, T, hc_mult, dim]`` stream.

    One extra launch, paid only where a caller genuinely needs the stream on its own.
    """
    shape = state.shape
    if state.stream is not None:
        return state.stream.view(shape)
    a, res, post, comb = state.pending
    M, hc, dim = shape[0] * shape[1], shape[2], shape[3]
    return hc_post_combine(a.reshape(M, dim), res.view(M, hc, dim), post, comb).view(shape)


__all__ = ["HCState", "hc_materialize", "UNFUSED_HC"]
