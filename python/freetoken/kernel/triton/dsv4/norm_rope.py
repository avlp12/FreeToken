"""Fused RMSNorm + decode RoPE for DeepSeek-V4 (decode fixed-cost fusion).

Every MLA decode layer runs the same three-step shape twice -- once for the query
and once for the latent KV:

    freqs_t = freqs_cis.index_select(0, pos)      # torch gather
    y = rms_norm(x, w, eps)                       # one Triton kernel
    rope_decode_inplace(y[..., -rope_dim:], freqs_t)   # one Triton kernel

and the compressor runs it a third time. That is six launches per layer for
q/kv/o (the ``o`` inverse rope shares the same gather), of which three are pure
dispatch: the gather moves ``rope_dim`` floats, and the rope kernel rewrites a
64-wide tail of a row the norm kernel had in registers a moment earlier.

``rms_norm_rope_decode`` does all of it in one program per row. The gather is
folded into the frequency load (the kernel indexes the full table by the row's
position); the rope tail is recomputed from a second, L1-resident load of the
same input row rather than round-tripping through HBM.

Bit-identity is by construction, and the construction has one subtlety worth
naming: the reference *stores* the normalised row as bf16 and the rope kernel
then re-loads and upcasts it, so the rope math sees a bf16-rounded operand. The
fused kernel reproduces that rounding explicitly (``.to(compute_type).to(fp32)``)
instead of carrying full fp32 precision into the rotation -- being more accurate
here would be a behaviour change, not an improvement. Everything else is the same
Triton reduction (same ``BLOCK_D``, same ``num_warps``, so the same reduction
tree) and the same fp32 complex multiply.

``FREETOKEN_UNFUSED_NORM_ROPE=1`` restores the three-step composition.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .norm import _TL


def unfused_norm_rope() -> bool:
    """``FREETOKEN_UNFUSED_NORM_ROPE=1`` -> take the rms_norm + index_select + rope chain."""
    return os.environ.get("FREETOKEN_UNFUSED_NORM_ROPE", "0") not in ("0", "")


@triton.jit
def _rmsnorm_rope_kernel(
    x_ptr, w_ptr, out_ptr, freqs_ptr, pos_ptr,
    M, D, RD, HEADS, eps,
    stride_xm, stride_om,
    stride_freq_pos, stride_freq_dim,
    BLOCK_D: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    HAS_W: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    compute_type: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    # Rows are (batch, head) in row-major order, and RoPE frequencies are per
    # BATCH row -- every head of a token shares its position.
    batch = row // HEADS

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    # Same expression, BLOCK_D and num_warps as `_rmsnorm_kernel`, hence the same
    # reduction tree -- this is a Triton-to-Triton identity, not an ATen one.
    var = tl.sum(x * x, axis=0) / D
    scale = tl.rsqrt(var + eps)
    y = x * scale
    if HAS_W:
        y = y * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Store everything EXCEPT the rope tail. The tail is written below by a
    # different set of lanes; masking it out here keeps the two stores disjoint,
    # so there is no intra-CTA write race and no ordering assumption between them.
    tl.store(out_ptr + row * stride_om + offs, y.to(compute_type),
             mask=mask & (offs < D - RD))

    # --- rope tail -------------------------------------------------------- #
    pair = tl.arange(0, PAIR_BLOCK)
    pmask = pair < (RD // 2)
    o_re = (D - RD) + pair * 2
    o_im = (D - RD) + pair * 2 + 1
    xr = tl.load(x_ptr + row * stride_xm + o_re, mask=pmask, other=0.0).to(tl.float32)
    xi = tl.load(x_ptr + row * stride_xm + o_im, mask=pmask, other=0.0).to(tl.float32)
    yr = xr * scale
    yi = xi * scale
    if HAS_W:
        yr = yr * tl.load(w_ptr + o_re, mask=pmask, other=0.0).to(tl.float32)
        yi = yi * tl.load(w_ptr + o_im, mask=pmask, other=0.0).to(tl.float32)
    # The reference rounds to the output dtype in rms_norm's store and the rope
    # kernel reloads that; round here too so the rotation sees the same operand.
    yr = yr.to(compute_type).to(tl.float32)
    yi = yi.to(compute_type).to(tl.float32)

    frow = tl.load(pos_ptr + batch)
    f_re = tl.load(freqs_ptr + frow * stride_freq_pos + pair * 2 * stride_freq_dim,
                   mask=pmask, other=0.0)
    f_im = tl.load(freqs_ptr + frow * stride_freq_pos + (pair * 2 + 1) * stride_freq_dim,
                   mask=pmask, other=0.0)
    if IS_INVERSE:
        out_re = yr * f_re + yi * f_im
        out_im = yi * f_re - yr * f_im
    else:
        out_re = yr * f_re - yi * f_im
        out_im = yr * f_im + yi * f_re
    tl.store(out_ptr + row * stride_om + o_re, out_re.to(compute_type), mask=pmask)
    tl.store(out_ptr + row * stride_om + o_im, out_im.to(compute_type), mask=pmask)


def rms_norm_rope_decode(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    *,
    heads: int = 1,
    inverse: bool = False,
) -> torch.Tensor:
    """``rope(rms_norm(x, weight, eps)[..., -rope_dim:], freqs_cis[positions])``.

    ``x``: ``[B, 1, (H,) D]``, last dim contiguous. ``freqs_cis``: the FULL
    ``[max_seq, rope_dim//2]`` complex table -- the kernel gathers. ``heads``: rows
    per batch element (``H``, or 1), so the kernel can map a flattened row back to
    the batch row whose position it must use.

    Returns a fresh tensor shaped like ``x``, exactly as ``rms_norm`` does; the
    rope is applied to it, not to ``x``.

    Fixed shapes and no host branch on device data, so the call is CUDA-graph
    capturable. ``positions`` is read on the device, so a captured graph rotates
    by whatever position the buffer holds at replay -- the same contract
    ``rope_decode_inplace`` already had.
    """
    D = x.shape[-1]
    assert rope_dim % 2 == 0 and 0 < rope_dim <= D
    x2d = x.reshape(-1, D)
    M = x2d.shape[0]
    # row -> batch is integer division by `heads`, and the batch index then indexes
    # `positions`. Getting `heads` wrong would not fail loudly inside the kernel --
    # it would read a position off the end of the buffer and rotate by garbage --
    # so check the row count against the position count here instead.
    assert M == heads * positions.numel(), (M, heads, positions.numel())
    out_dtype = x.dtype if x.dtype in _TL else torch.bfloat16
    out = torch.empty_like(x2d, dtype=out_dtype)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    BLOCK_D = triton.next_power_of_2(D)
    # num_warps must match norm.rms_norm: it is what fixes the reduction tree, and
    # the reduction tree is what makes this bit-identical rather than merely close.
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    _rmsnorm_rope_kernel[(M,)](
        x2d, weight, out, freqs_real, positions,
        M, D, rope_dim, heads, eps,
        x2d.stride(0), out.stride(0),
        freqs_real.stride(0), freqs_real.stride(1),
        BLOCK_D=BLOCK_D,
        PAIR_BLOCK=triton.next_power_of_2(rope_dim // 2),
        HAS_W=weight is not None,
        IS_INVERSE=inverse,
        compute_type=_TL[out_dtype],
        num_warps=num_warps,
    )
    return out.reshape(x.shape)


__all__ = ["rms_norm_rope_decode", "unfused_norm_rope"]
