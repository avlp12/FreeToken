# SPDX-License-Identifier: Apache-2.0
"""UVA row gather for the Qwen3.8-Flash-Next PLE n-gram table -- IQ4_NL variant.

Upstream reference (``/root/ft-upstream/python/freetoken/kernel/triton/ple.py``,
``PinnedUVATable`` in ``models/qwen4_exp/ple.py``) keeps the 47.7 GiB PLE table
(FP8-e4m3 + one scalar scale) in pinned host memory and gathers rows over UVA with
one Triton program per requested row: read the row, widen to fp32, apply the
per-tensor scale, store bf16.

This fork's PLE table is sourced from the GGUF release instead, quantized GGML
IQ4_NL (26.82 GiB total for the same 320,001,536 x 160 table: 90 bytes/row instead
of 1 fp8 byte/element -- see ``freetoken.models.qwen4_exp.ngram``'s module docstring
for how that source was verified against the real GGUF). IQ4_NL has no single
per-tensor scale to multiply by; each row is 5 independent blocks, each with its
own fp16 scale and a 4-bit-per-element LUT code, so the upstream kernel's
"load raw, multiply by one scalar" body does not apply here -- this module adapts
the SAME pattern (pinned host table, one program per row, UVA pointer
reconstruction) to that block layout instead.

Block spec (ggml / llama.cpp, mirrored from this repo's own
``freetoken/kernel/csrc/gguf/ggml-common.h`` + ``dequantize.cuh``, and duplicated in
pure Python/torch by ``freetoken.models.qwen4_exp.ngram._dequantize_iq4nl_rows`` --
the mmap path's dequantizer this kernel's output must match bit-for-bit):

    struct block_iq4_nl { half d; uint8_t qs[16]; };  // 18 bytes / 32 elements
    y[k]    = d * kvalues_iq4nl[qs[k] & 0xf]   for k in [0, 16)
    y[k+16] = d * kvalues_iq4nl[qs[k] >> 4]    for k in [0, 16)

head_dim=160 is 5 independent blocks (5*18 = 90 bytes/row, matching the GGUF
reader's measured row_bytes=90). Per output element ``e`` in ``[0, head_dim)``:
block = e // 32, within = e % 32, is_hi = within >= 16,
nib_pos = within - 16 if is_hi else within (this is the index into the block's
16-byte ``qs``), code = (qs[nib_pos] >> 4) if is_hi else (qs[nib_pos] & 0xf).
``tests/kernel/test_ple_iq4nl_gather.py`` transcribes exactly this per-element
formula as an unvectorized scalar Python loop (no tensor-shape trick to hide an
off-by-one) and checks it byte-for-byte against ``_dequantize_iq4nl_rows`` on
random synthetic rows -- that is the formula this kernel implements via UVA
pointer arithmetic below, and is as far as a GPU-less test can verify it; the
kernel launch itself needs a live CUDA device (see that test file's module
docstring for exactly what the advisor should run on a GPU box to close the gap).

Ids outside the table store zeros. Latency-bound over PCIe/host memory like the
upstream kernel, so num_warps stays at 1 (many small requests in flight beats wide
warps here).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# GGML kvalues_iq4nl LUT (16 entries), identical to
# freetoken.models.qwen4_exp.ngram._KVALUES_IQ4NL -- kept in sync by
# tests/kernel/test_ple_iq4nl_gather.py::test_lut_matches_ngram_dequantizer.
KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

_BLOCK_ELEMS = 32  # QK4_NL
_BLOCK_BYTES = 18  # sizeof(block_iq4_nl) = sizeof(half) + 16
_NUM_WARPS = 1


@triton.jit
def _ple_gather_iq4nl_kernel(
    table_ptr,      # int64 address of the pinned host table's row 0, byte 0 --
                     # freetoken.kernel.pinned.device_ptr(), not necessarily
                     # the host tensor's data_ptr() (WDDM maps registered host
                     # memory to a different device address; Linux/UVA the two
                     # coincide -- see kernel/pinned.py:device_ptr).
    ids_ptr,        # row_ids, flat, one per program
    out_ptr,        # [n, HEAD_DIM], on `device`; store casts to its dtype
    lut_ptr,        # KVALUES_IQ4NL as a float32[16] CUDA tensor, same device as out
    num_rows,
    ROW_BYTES: tl.constexpr,     # (HEAD_DIM // BLOCK_ELEMS) * BLOCK_BYTES == 90
    HEAD_DIM: tl.constexpr,      # 160
    BLOCK_ELEMS: tl.constexpr,   # 32
    BLOCK_BYTES: tl.constexpr,   # 18
    BLOCK_D: tl.constexpr,       # next_power_of_2(HEAD_DIM)
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)

    # Scalar per-program byte address of this row's first byte in the pinned host
    # table, reconstructed as two differently-typed pointers over the SAME address
    # (uint8 for the packed qs nibbles, float16 for the per-block scale) -- the
    # same "rebuild a typed pointer from a raw host address" idiom the upstream
    # fp8/bf16 kernel uses (table_ptr.to(tl.int64).to(tl.pointer_type(...))), just
    # with a per-row BYTE stride instead of an ELEMENT stride.
    row_byte_base = table_ptr.to(tl.int64) + idx * ROW_BYTES
    u8_ptr = row_byte_base.to(tl.pointer_type(tl.uint8))
    f16_ptr = row_byte_base.to(tl.pointer_type(tl.float16))

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < HEAD_DIM
    block = offsets // BLOCK_ELEMS
    within = offsets % BLOCK_ELEMS
    is_hi = within >= (BLOCK_ELEMS // 2)
    nib_pos = tl.where(is_hi, within - BLOCK_ELEMS // 2, within)

    # qs[nib_pos] of block `block` -- qs starts 2 bytes into the block (after the
    # fp16 `d`); see the module docstring's block spec.
    qs_byte_offset = block * BLOCK_BYTES + 2 + nib_pos
    qbyte = tl.load(u8_ptr + qs_byte_offset, mask=mask, other=0)
    code = tl.where(is_hi, (qbyte >> 4) & 0x0F, qbyte & 0x0F).to(tl.int32)

    # block_iq4_nl.d (half): one every BLOCK_BYTES=18 bytes == 9 float16 elements
    # (18 is even, so every block boundary stays float16-aligned relative to the
    # row's byte base).
    d_f16_offset = block * (BLOCK_BYTES // 2)
    d = tl.load(f16_ptr + d_f16_offset, mask=mask, other=0.0).to(tl.float32)

    lut_val = tl.load(lut_ptr + code, mask=mask, other=0.0)
    values = tl.where(in_range, lut_val * d, 0.0)
    tl.store(
        out_ptr + row * HEAD_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_rows_iq4nl(
    table_ptr: int,
    num_rows: int,
    head_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Gather IQ4_NL rows from the pinned host table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), head_dim]``
    on the same device (any float dtype -- the kernel store casts). ``lut`` is
    ``KVALUES_IQ4NL`` as a float32 CUDA tensor (16 entries), same device as ``out``.
    ``table_ptr`` is the address the GPU must dereference
    (``freetoken.kernel.pinned.device_ptr``), not necessarily the host table
    tensor's ``data_ptr()``.
    """
    assert head_dim % _BLOCK_ELEMS == 0, head_dim
    n = row_ids.numel()
    assert out.shape == (n, head_dim) and out.is_contiguous(), out.shape
    row_bytes = (head_dim // _BLOCK_ELEMS) * _BLOCK_BYTES
    if n:
        _ple_gather_iq4nl_kernel[(n,)](
            table_ptr,
            row_ids,
            out,
            lut,
            num_rows,
            ROW_BYTES=row_bytes,
            HEAD_DIM=head_dim,
            BLOCK_ELEMS=_BLOCK_ELEMS,
            BLOCK_BYTES=_BLOCK_BYTES,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["KVALUES_IQ4NL", "ple_gather_rows_iq4nl"]
