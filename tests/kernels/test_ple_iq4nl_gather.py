"""CPU-only numerical proof for the PLE IQ4_NL UVA gather kernel's dequant formula.

``freetoken.kernel.triton.ple._ple_gather_iq4nl_kernel`` needs a live CUDA device to
launch (it dereferences pinned host memory over UVA), so it cannot be exercised
directly without a GPU -- see that module's docstring and this file's
``test_ple_gather_rows_iq4nl_launches_the_real_kernel_and_matches_mmap`` (skipped
here, for the advisor to run on a GPU box).

What CAN be checked without a GPU, and what these tests check, is that the
kernel's per-element addressing formula (block index, within-block position,
hi/lo nibble, byte offsets into ``qs``, the ``d`` scale read) is exactly the block
spec both the kernel and ``freetoken.models.qwen4_exp.ngram._dequantize_iq4nl_rows``
(the existing, unmodified mmap-path dequantizer) implement. ``_scalar_iq4nl_row``
below is a deliberately unvectorized, element-at-a-time transcription of that
formula -- no tensor-shape trick that could silently paper over an off-by-one in
the block/byte-offset arithmetic the Triton kernel performs via pointer math. If
this matches ``_dequantize_iq4nl_rows`` bit-for-bit on random rows, the formula the
kernel was translated from is correct; the translation into Triton pointer
arithmetic itself is what the advisor's GPU run in
``test_ple_gather_rows_iq4nl_launches_the_real_kernel_and_matches_mmap`` closes.
"""

from __future__ import annotations

import struct

import pytest
import torch

from freetoken.kernel.triton.ple import KVALUES_IQ4NL, _BLOCK_BYTES, _BLOCK_ELEMS, ple_gather_rows_iq4nl
from freetoken.models.qwen4_exp.ngram import (
    _IQ4NL_BLOCK_BYTES,
    _IQ4NL_BLOCK_ELEMS,
    _KVALUES_IQ4NL,
    _dequantize_iq4nl_rows,
)

HEAD_DIM = 160  # released checkpoint's real head_dim: 5 blocks of 32
N_BLOCKS = HEAD_DIM // _BLOCK_ELEMS
ROW_BYTES = N_BLOCKS * _BLOCK_BYTES


def test_lut_matches_ngram_dequantizer():
    """The kernel's LUT and the mmap path's LUT must be the literal same 16 values
    (see freetoken/kernel/triton/ple.py's KVALUES_IQ4NL comment) -- a silent edit to
    either one, without the other, would desync bit-for-bit output between the two
    gather paths without any shape/type error to catch it."""
    assert _BLOCK_ELEMS == _IQ4NL_BLOCK_ELEMS
    assert _BLOCK_BYTES == _IQ4NL_BLOCK_BYTES
    assert KVALUES_IQ4NL == _KVALUES_IQ4NL.tolist()


def _random_iq4nl_row_bytes(rng: torch.Generator, n_rows: int) -> torch.Tensor:
    """``[n_rows, ROW_BYTES]`` uint8, structurally valid block_iq4_nl rows.

    ``qs`` (the packed 4-bit codes) has no invalid byte pattern, so those are
    uniform random bytes. ``d`` (the per-block fp16 scale) is built from an
    actual finite float16 draw instead of raw random bytes: a real checkpoint's
    scales are always finite, and raw-random bytes occasionally land on fp16's
    NaN/Inf bit patterns (exponent all-1s), which both dequantizers reproduce
    correctly but which then makes ``torch.testing.assert_close`` report a
    mismatch on a NaN==NaN comparison -- a test artifact, not a formula bug."""
    n_blocks = ROW_BYTES // _BLOCK_BYTES
    d = (torch.rand(n_rows, n_blocks, generator=rng) * 4 - 2).to(torch.float16)
    d_bytes = d.contiguous().view(torch.uint8).view(n_rows, n_blocks, 2)
    qs = torch.randint(
        0, 256, (n_rows, n_blocks, _BLOCK_BYTES - 2), dtype=torch.uint8, generator=rng
    )
    blocks = torch.cat([d_bytes, qs], dim=-1)
    return blocks.reshape(n_rows, ROW_BYTES).contiguous()


def _scalar_iq4nl_row(raw_row: bytes, head_dim: int) -> list[float]:
    """Unvectorized, element-at-a-time mirror of ``_ple_gather_iq4nl_kernel``'s
    addressing formula (see freetoken/kernel/triton/ple.py's module docstring for
    the derivation) -- deliberately not sharing any code with
    ``_dequantize_iq4nl_rows``'s block-vectorized approach, so a bug shared by both
    implementations of the SAME formula would still show up as a mismatch here
    only if the two formulas actually diverge, not if they're both wrong the same
    way from copy-paste."""
    out = []
    for elem in range(head_dim):
        block = elem // _BLOCK_ELEMS
        within = elem % _BLOCK_ELEMS
        is_hi = within >= (_BLOCK_ELEMS // 2)
        nib_pos = within - _BLOCK_ELEMS // 2 if is_hi else within
        block_off = block * _BLOCK_BYTES
        d = struct.unpack("<e", raw_row[block_off : block_off + 2])[0]
        qbyte = raw_row[block_off + 2 + nib_pos]
        code = (qbyte >> 4) if is_hi else (qbyte & 0x0F)
        out.append(float(KVALUES_IQ4NL[code]) * float(d))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 12345])
def test_scalar_per_element_formula_matches_dequantize_iq4nl_rows(seed: int):
    rng = torch.Generator().manual_seed(seed)
    n_rows = 6
    raw = _random_iq4nl_row_bytes(rng, n_rows)

    reference = _dequantize_iq4nl_rows(raw, HEAD_DIM, torch.float32)
    assert reference.shape == (n_rows, HEAD_DIM)

    for r in range(n_rows):
        row_bytes = raw[r].numpy().tobytes()
        scalar = torch.tensor(_scalar_iq4nl_row(row_bytes, HEAD_DIM), dtype=torch.float32)
        torch.testing.assert_close(scalar, reference[r], rtol=0, atol=0)


def test_scalar_reference_is_sensitive_to_a_single_bit_flip():
    """Sanity check on the test itself: if the two formulas agreed by both being
    insensitive to the input (e.g. a shape bug reading all zeros), the equality
    above would pass vacuously. Confirm a single flipped byte actually changes the
    output, so the match above is a real formula check, not a degenerate one."""
    rng = torch.Generator().manual_seed(7)
    raw = _random_iq4nl_row_bytes(rng, 1)
    base = _scalar_iq4nl_row(raw[0].numpy().tobytes(), HEAD_DIM)

    flipped = raw.clone()
    flipped[0, 2] ^= 0xFF  # flip a qs byte (block 0's first nibble pair)
    changed = _scalar_iq4nl_row(flipped[0].numpy().tobytes(), HEAD_DIM)
    assert base != changed


def test_ple_gather_rows_iq4nl_empty_is_a_noop_and_needs_no_cuda():
    """n=0 must short-circuit before the Triton launch (see ple_gather_rows_iq4nl's
    `if n:` guard) -- this is the one call shape this test can make into the real
    wrapper without a GPU."""
    row_ids = torch.empty(0, dtype=torch.int64)
    out = torch.empty(0, HEAD_DIM, dtype=torch.bfloat16)
    lut = torch.tensor(KVALUES_IQ4NL, dtype=torch.float32)

    result = ple_gather_rows_iq4nl(0, 0, HEAD_DIM, row_ids, out, lut)
    assert result is out
    assert result.numel() == 0


def test_ple_gather_rows_iq4nl_rejects_mismatched_output_shape_before_any_launch():
    """The shape assert runs before the `if n:` launch guard, so this is checkable
    without CUDA too: a caller bug (wrong head_dim, non-contiguous out) must fail
    loudly instead of silently launching a kernel that will scribble OOB."""
    row_ids = torch.empty(0, dtype=torch.int64)
    wrong_shape_out = torch.empty(0, HEAD_DIM + 1, dtype=torch.bfloat16)
    lut = torch.tensor(KVALUES_IQ4NL, dtype=torch.float32)

    with pytest.raises(AssertionError):
        ple_gather_rows_iq4nl(0, 0, HEAD_DIM, row_ids, wrong_shape_out, lut)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a live CUDA device for UVA")
def test_ple_gather_rows_iq4nl_launches_the_real_kernel_and_matches_mmap():
    """GPU-only: build a small synthetic IQ4_NL table, pin it, gather a batch of
    ids through the real Triton kernel, and compare against
    ``_dequantize_iq4nl_rows`` applied to the same rows (the mmap path's own
    dequantizer). This is the check this file cannot do without a GPU -- see the
    module docstring. Advisor: run with
    ``PYTHONPATH=/root/ft-ple/python pytest tests/kernels/test_ple_iq4nl_gather.py -k launches_the_real_kernel -v``
    on the GPU box.
    """
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr

    device = torch.device("cuda", torch.cuda.current_device())
    rng = torch.Generator().manual_seed(42)
    num_rows = 1000
    raw = _random_iq4nl_row_bytes(rng, num_rows)
    pinned = alloc_pinned_tensor(num_rows, ROW_BYTES, dtype=torch.uint8)
    pinned.copy_(raw)
    table_ptr = device_ptr(pinned)

    row_ids = torch.randint(0, num_rows, (37,), dtype=torch.int64, device=device)
    out = torch.empty(37, HEAD_DIM, dtype=torch.bfloat16, device=device)
    lut = torch.tensor(KVALUES_IQ4NL, dtype=torch.float32, device=device)
    ple_gather_rows_iq4nl(table_ptr, num_rows, HEAD_DIM, row_ids, out, lut)

    expected_rows = _dequantize_iq4nl_rows(
        raw[row_ids.cpu()], HEAD_DIM, torch.float32
    ).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected_rows, rtol=0, atol=0)
