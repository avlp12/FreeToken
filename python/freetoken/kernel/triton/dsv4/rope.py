"""Fused interleaved RoPE for DeepSeek-V4 decode (borrowed from sglang's Triton rope).

FreeToken's decode rope was a chain of torch ops (``view_as_complex`` -> complex mul ->
``view_as_real`` -> ``copy_``), ~3-5 small ``at::native`` kernels per call, called for q/kv/o
in every layer. This collapses each call into ONE ``@triton.jit`` kernel.

The math is identical to ``ops.apply_rotary_emb_decode``: pairing the interleaved last
``rope_dim`` of ``x`` as ``(real, imag)`` and multiplying by the per-row complex ``freqs``
(``inverse`` uses the conjugate). All compute in fp32, stored back to ``x``'s dtype -- so it is
bit-identical to the torch path (modulo a possible 1-ULP FMA, gated by the decode parity check).

Kernel adapted from sglang ``srt/layers/deepseek_v4_rope.py:apply_rotary_emb_triton_kernel``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_decode_kernel(
    x_ptr, freqs_ptr, pos_ptr,
    rope_dim,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_freq_pos, stride_freq_dim,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    HAS_POS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_dim = tl.program_id(2)

    # HAS_POS: index the FULL frequency table by this row's position, instead of
    # taking a table the caller pre-gathered with index_select. Same table entries,
    # so the arithmetic is untouched -- it just moves the gather off the launch
    # queue, which is the whole cost of it at one decode row.
    freq_row = tl.load(pos_ptr + pid_batch) if HAS_POS else pid_batch

    base = pid_batch * stride_x_batch + (pid_head * stride_x_head if IS_3D else 0)
    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_pair < (rope_dim // 2)

    offs_real = base + offs_pair * 2 * stride_x_dim
    offs_imag = base + (offs_pair * 2 + 1) * stride_x_dim
    x_real = tl.load(x_ptr + offs_real, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + offs_imag, mask=mask, other=0.0).to(tl.float32)

    f_off_real = freq_row * stride_freq_pos + offs_pair * 2 * stride_freq_dim
    f_off_imag = freq_row * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim
    f_real = tl.load(freqs_ptr + f_off_real, mask=mask, other=0.0)
    f_imag = tl.load(freqs_ptr + f_off_imag, mask=mask, other=0.0)

    if IS_INVERSE:  # multiply by conj(freqs)
        out_real = x_real * f_real + x_imag * f_imag
        out_imag = x_imag * f_real - x_real * f_imag
    else:
        out_real = x_real * f_real - x_imag * f_imag
        out_imag = x_real * f_imag + x_imag * f_real

    tl.store(x_ptr + offs_real, out_real, mask=mask)
    tl.store(x_ptr + offs_imag, out_imag, mask=mask)


def rope_decode_inplace(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place interleaved RoPE on the last ``rope_dim`` of ``x``, per-row complex ``freqs_cis``.

    ``x``: ``[B, 1, (H,) rope_dim]`` (one decode token per row; rope applies to the last dim, which
    may be a non-contiguous slice -- strides are honored). Mutates and returns ``x``.

    ``freqs_cis``: ``[B, rope_dim//2]`` complex, one row per decode row -- OR, when
    ``positions`` (int64 ``[B]``) is given, the whole ``[max_seq, rope_dim//2]`` table,
    which the kernel indexes itself. The gathered and the folded form read the same
    table entries and do the same arithmetic; folding just deletes the caller's
    ``freqs_cis.index_select(0, pos)`` launch, which at one decode row costs more to
    dispatch than to execute. Three rope calls per layer share one gather today, so
    folding saves one launch per layer, not three.

    One thing the folded form gives up: ``index_select`` bounds-checks its index on
    the device, this kernel does not. A position past ``max_seq_len`` read garbage
    frequencies instead of tripping an assert. Positions come from the scheduler
    already clamped to the allocated context, so this is a lost backstop rather
    than a live hazard -- but it is the reason to keep the escape hatch honest.
    """
    xv = x.squeeze(1)  # drop the seq==1 dim -> [B, (H,) rope_dim]
    is_3d = xv.ndim == 3
    if is_3d:
        B, H, rope_dim = xv.shape
    else:
        B, rope_dim = xv.shape
        H = 1
    # view_as_real + flatten on a contiguous complex tensor is a pure view either way
    # (gathered [B, rd] row or the full [max_seq, rd] table), so contiguous() is free.
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    grid = (B, H, triton.cdiv(rope_dim // 2, 128))
    _rope_decode_kernel[grid](
        xv, freqs_real, positions if positions is not None else freqs_real,
        rope_dim,
        xv.stride(0), xv.stride(1) if is_3d else 0, xv.stride(-1),
        freqs_real.stride(0), freqs_real.stride(1),
        IS_INVERSE=inverse, IS_3D=is_3d, HAS_POS=positions is not None,
        BLOCK_SIZE=128,
    )
    return x


__all__ = ["rope_decode_inplace"]
