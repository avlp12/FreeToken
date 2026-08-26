"""Fused clamped-SwiGLU for the DeepSeek-V4 shared (dense) expert.

Reference does ``silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`` in fp32
(``w1(x).float()``, ``w3(x).float()``) then casts back to bf16 -- that's 2 upcasts,
clamp x2, silu, mul, downcast = 7 launches/layer x 43 layers. This kernel reads the
bf16 gate/up directly, does the math in fp32 internally, and writes bf16: bit-exact
to the reference (the inputs were exact bf16 upcasts) in a single launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.pdl import gdc_launch_dependents, gdc_wait, pdl_enabled

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, N, limit, BLOCK: tl.constexpr,
                   HAS_LIMIT: tl.constexpr, OUT: tl.constexpr,
                   ENABLE_PDL: tl.constexpr = False):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # gate/up are both produced by the two projections that ran just before us.
    if ENABLE_PDL:
        gdc_wait()
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        u = tl.minimum(tl.maximum(u, -limit), limit)
        g = tl.minimum(g, limit)
    g = g * tl.sigmoid(g)
    tl.store(out_ptr + offs, (g * u).to(OUT), mask=mask)
    if ENABLE_PDL:
        gdc_launch_dependents()


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float,
                 out_dtype: torch.dtype) -> torch.Tensor:
    """``silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`` -> ``out_dtype``."""
    out = torch.empty_like(gate, dtype=out_dtype)
    N = gate.numel()
    BLOCK = 512
    pdl = pdl_enabled()
    _swiglu_kernel[(triton.cdiv(N, BLOCK),)](
        gate, up, out, N, float(limit), BLOCK=BLOCK,
        HAS_LIMIT=limit > 0, OUT=_TL[out_dtype], ENABLE_PDL=pdl, launch_pdl=pdl,
        num_warps=4,
    )
    return out


__all__ = ["fused_swiglu"]
