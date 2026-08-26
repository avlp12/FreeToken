"""fast_index_copy_multi's UNPITCHED path with a bank row that is not 16-byte aligned.

Bring-up bug: Qwen3.8-Flash-Next's fp8_block expert banks carry two per-block
``weight_scale_inv`` scale planes -- gate_up_scale at (2*I//128, H//128) bf16 =
(10, 20) = 400 B/expert, down_scale at (H//128, I//128) bf16 = (20, 5) = 200 B/expert
(H=2560, I=640). 400 is a multiple of 16; 200 is not (200 % 16 == 8). The single-bank
compile-time-templated kernel (fast_index_copy_jit) has NO legal instantiation for
either width (see kernel/fast_index_copy.py's _shrink_worker_feature_size and
/root/qwen4_gguf_probe/r4.py), and until this fix OffloadMoeCache._build_copy_plan
required every bank's row bytes to be a 16-byte multiple to enable the runtime-sized
multi-bank path at all -- so down_scale's 200 B row silently disabled the FUSED copy
for every fp8_block bank (not just itself), and copy_missing() fell back to the
single-bank kernel, which then failed to JIT-compile on gate_up_scale's 400 B row
(the first bank in schema order) with an nvcc static_assert 30+ seconds into boot.

The fix: fast_index_copy_multi's unpitched branch now moves a bank row with 8-byte
(uint2) units instead of 16-byte (uint4) ones whenever the row isn't 16-aligned ("=
narrow8" below). This is exact, not a truncated copy: every fp8_block scale-plane
width divides 8 (both 400 and 200 do), and an 8-byte-aligned feat keeps every expert
row's start address 8-byte aligned regardless of the expert index (ps*feat % 8 == 0
for any integer ps), so uint2 loads/stores stay correctly aligned across the whole
bank -- see the narrow8 branch in fast_index_copy_multi (fast_index_copy.cuh) and the
``feat % 8`` eligibility check in OffloadMoeCache._build_copy_plan (offload_cache.py).

This test builds banks at exactly those two shapes (plus a normal 16-aligned weight
bank in the SAME launch, matching how copy_missing() always copies gate_up,
gate_up_scale, down, down_scale together) and asserts the destination is bit-identical
to a reference torch gather -- not just "did not crash".
"""

from __future__ import annotations

import torch

import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_EXPERTS = 512
CACHE_SIZE = 48  # << NUM_EXPERTS, so poison-tail / unaddressed-slot checks mean something
POISON = 0xCD

# Real Qwen3.8-Flash-Next fp8_block geometry (H=2560, I=640, 128x128 scale blocks).
GATE_UP_SCALE_SHAPE = (10, 20)  # bf16, 400 B/expert -- already 16-aligned
DOWN_SCALE_SHAPE = (20, 5)      # bf16, 200 B/expert -- the narrow8 case (200 % 16 == 8)
# A stand-in dense weight row (fp8_e4m3, 1 byte/elem): large and cleanly 16-aligned,
# exercising the SAME launch's fast (uint4) branch alongside the narrow8 (uint2) one.
GATE_UP_SHAPE = (1280, 2560)


def _i64(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64, device="cuda")


class _Bank:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, seed: int):
        g = torch.Generator().manual_seed(seed)
        if dtype.is_floating_point:
            self.src = torch.randn((NUM_EXPERTS, *shape), generator=g, dtype=torch.float32).to(dtype).cuda()
        else:
            self.src = torch.randint(
                0, 256, (NUM_EXPERTS, *shape), generator=g, dtype=torch.uint8
            ).cuda()
        # Allocate native, then poison through a byte-view alias: uint8->wider-dtype
        # views require an even last dim (fails for e.g. (.., 5) bf16), but the
        # reverse (wider dtype -> uint8) always works on a contiguous tensor.
        self.dst = torch.empty((CACHE_SIZE, *shape), dtype=dtype, device="cuda")
        self.dst.view(torch.uint8).fill_(POISON)
        self.dst_bytes = self.dst.view(torch.uint8).view(CACHE_SIZE, -1)
        self.feat_bytes = shape_bytes(shape, dtype)


def shape_bytes(shape, dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * torch.empty((), dtype=dtype).element_size()


def _launch(banks: list[_Bank], dst_idx, src_idx, num=None, bpb=8):
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

    dst_ptrs = _i64(b.dst.data_ptr() for b in banks)
    src_ptrs = _i64(b.src.data_ptr() for b in banks)
    feat_bytes = _i64(b.feat_bytes for b in banks)
    fast_index_copy_multi_jit(
        dst_ptrs, src_ptrs, feat_bytes, dst_idx, src_idx, num, blocks_per_bank=bpb,
    )
    torch.cuda.synchronize()


def _indices(n: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    src = torch.randperm(NUM_EXPERTS, generator=g, device="cuda")[:n].to(torch.int32)
    dst = torch.randperm(CACHE_SIZE, generator=g, device="cuda")[:n].to(torch.int32)
    return dst.contiguous(), src.contiguous()


def _reference_gather(bank: _Bank, dst_idx, src_idx) -> torch.Tensor:
    """What a plain torch index_select/index_copy would produce -- the independent
    reference the assignment asks for, computed without touching the custom kernel."""
    out = bank.dst.clone()
    out[dst_idx.long()] = bank.src[src_idx.long()]
    return out


def _make_banks() -> list[_Bank]:
    return [
        _Bank(GATE_UP_SHAPE, torch.float8_e4m3fn, seed=1),   # fast (uint4) path, feat % 16 == 0
        _Bank(GATE_UP_SCALE_SHAPE, torch.bfloat16, seed=2),  # feat=400, % 16 == 0 (uint4 path)
        _Bank(DOWN_SCALE_SHAPE, torch.bfloat16, seed=3),     # feat=200, % 16 != 0 (narrow8 path)
    ]


def test_bank_feat_bytes_match_the_qwen4_exp_checkpoint():
    banks = _make_banks()
    assert banks[1].feat_bytes == 400
    assert banks[2].feat_bytes == 200
    assert banks[1].feat_bytes % 16 == 0, "gate_up_scale should already take the fast uint4 path"
    assert banks[2].feat_bytes % 16 != 0, "down_scale must actually exercise the narrow8 path"
    assert banks[2].feat_bytes % 8 == 0, "narrow8 requires 8-byte alignment"


def test_narrow_scale_row_is_bit_identical_to_torch_gather():
    """The core correctness claim: copying a 200 B/expert bank alongside 16-aligned
    banks in ONE fused launch reproduces exactly what independent torch indexing does."""
    n = 20
    dst_idx, src_idx = _indices(n, seed=101)
    banks = _make_banks()

    refs = [_reference_gather(b, dst_idx, src_idx) for b in banks]
    _launch(banks, dst_idx, src_idx)

    for i, (b, ref) in enumerate(zip(banks, refs)):
        assert torch.equal(b.dst, ref), f"bank {i} (feat={b.feat_bytes} B) diverged from the torch reference"

    # Explicitly re-check the addressed rows byte-for-byte against the SOURCE too (not
    # just against the reference tensor), for the narrow8 bank specifically.
    narrow = banks[2]
    got = narrow.dst[dst_idx.long()].view(torch.uint8)
    want = narrow.src[src_idx.long()].view(torch.uint8)
    assert torch.equal(got, want), "down_scale (200 B/expert) payload is not bit-identical"


def test_narrow_scale_leaves_unaddressed_slots_and_no_tail_corruption():
    n = 15
    dst_idx, src_idx = _indices(n, seed=202)
    banks = _make_banks()
    _launch(banks, dst_idx, src_idx)

    narrow = banks[2]
    touched = torch.zeros(CACHE_SIZE, dtype=torch.bool, device="cuda")
    touched[dst_idx.long()] = True
    untouched_bytes = narrow.dst_bytes[~touched]
    assert (untouched_bytes == POISON).all(), (
        "narrow8 (200 B/expert) wrote outside its addressed rows -- units/stride "
        "arithmetic is reading past the intended row"
    )
    # Every addressed row is FULLY overwritten (200 B is exactly 25 whole 8-byte
    # units, no remainder to drop) -- checked byte-for-byte against the real source
    # in test_narrow_scale_row_is_bit_identical_to_torch_gather. A "no leftover
    # POISON byte" check here would be unsound on its own: real bf16 data can
    # legitimately contain a byte equal to the poison value by chance.


def test_narrow_scale_alone_single_bank_launch():
    """Same as above but with the narrow bank as the ONLY bank in the launch (rules out
    the fix accidentally depending on being alongside a fast-path bank)."""
    n = 10
    dst_idx, src_idx = _indices(n, seed=303)
    banks = [_Bank(DOWN_SCALE_SHAPE, torch.bfloat16, seed=4)]
    ref = _reference_gather(banks[0], dst_idx, src_idx)
    _launch(banks, dst_idx, src_idx)
    assert torch.equal(banks[0].dst, ref)


@pytest.mark.parametrize("bpb", [1, 8, 64])
def test_narrow_scale_grid_width_does_not_change_the_result(bpb):
    n = 13
    dst_idx, src_idx = _indices(n, seed=404)
    banks = _make_banks()
    refs = [_reference_gather(b, dst_idx, src_idx) for b in banks]
    _launch(banks, dst_idx, src_idx, bpb=bpb)
    for b, ref in zip(banks, refs):
        assert torch.equal(b.dst, ref)


def test_narrow_scale_respects_valid_length_truncation():
    n, live = 14, 6
    dst_idx, src_idx = _indices(n, seed=505)
    num = torch.tensor([live], dtype=torch.int64, device="cuda")
    banks = _make_banks()

    live_dst, live_src = dst_idx[:live], src_idx[:live]
    refs = [_reference_gather(b, live_dst, live_src) for b in banks]
    _launch(banks, dst_idx, src_idx, num=num)
    for b, ref in zip(banks, refs):
        assert torch.equal(b.dst, ref), "num_indices truncation not honored by the narrow8 path"
