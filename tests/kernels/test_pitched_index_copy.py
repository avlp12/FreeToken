"""Payload-only weight rows in the fused multi-bank index copy.

``fast_index_copy_multi`` moves whole bank rows between a host bank and the GPU slot
cache, and a bank row is one EXPERT: a stack of weight rows at a uniform byte pitch.
That pitch has until now been the transfer size as well. For a mixed-quant bank the
two are not the same number. The q2_k_ud down bank pitches every weight row to 1088 B
so the two MXFP4 layers fit, but the other 41 layers hold 784 B IQ3_XXS rows; copying
the 304 B tail of each of 4096 rows is 28% of that bank's PCIe traffic buying nothing,
because the GEMV's inner loop runs ``blocks_per_row`` iterations of the type it was
CALLED with and never touches the tail.

So ``copy_bytes`` (payload per weight row) and ``row_pitch`` (weight-row stride) split
the two jobs, while the expert stride stays ``feat_bytes`` on both sides. What has to
be true:

* with them absent the kernel is byte-for-byte what it was -- this is the hottest
  kernel in the decode loop, so "absent" and "present but equal to the pitch" are both
  checked against a full-pitch reference;
* every weight row's prefix is exact and every tail is left ALONE, not zeroed. The
  whole argument is that those bytes are unread; a kernel that helpfully cleared them
  would be moving the traffic again. This is the test that would catch treating the
  payload as one contiguous prefix of the expert (which copies whole weight rows and
  leaves the rest of them stale) instead of a prefix of each row;
* the saving is real -- and it is NOT automatic. Fewer bytes does not mean less time:
  the source is pinned host memory read over PCIe, and skipping a sub-page gap in
  every row breaks the read into shorter runs. Measured on this box, 1184-of-1568
  (gate_up) costs 79% of the full-pitch time for 76% of the bytes, while 784-of-1088
  (down) costs 150% of the time for 72% of the bytes. Both numbers are asserted, the
  second deliberately as an expected LOSS, so the policy that narrows only the
  gate_up bank has a test that fails if the hardware ever changes its mind.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Two banks with the shipped q2_k_ud geometry, scaled down in row count:
# gate_up pitch 1568 B (IQ3_XXS at H=4096) holding 1184 B IQ2_XS rows,
# down pitch 1088 B (MXFP4 at I=2048) holding 784 B IQ3_XXS rows.
GU_PITCH, GU_NATIVE = 1568, 1184
DN_PITCH, DN_NATIVE = 1088, 784

NUM_EXPERTS = 32
SLOTS = 24
GU_ROWS = 64
DN_ROWS = 48

POISON = 0xAB


def _i64(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64, device="cuda")


class _Banks:
    """Two banks plus their slot caches, laid out the way OffloadMoeCache does it."""

    pitches = (GU_PITCH, DN_PITCH)

    rows = (GU_ROWS, DN_ROWS)
    slots = SLOTS

    def __init__(self, seed: int = 0):
        shapes = [(NUM_EXPERTS, r, p) for r, p in zip(self.rows, self.pitches)]
        g = torch.Generator().manual_seed(seed)
        self.src = [
            torch.randint(0, 256, s, generator=g, dtype=torch.uint8).cuda() for s in shapes
        ]
        # Poison the caches so "the tail was left alone" has something visible to be.
        self.dst = [
            torch.full((SLOTS, *s[1:]), POISON, dtype=torch.uint8, device="cuda") for s in shapes
        ]
        self.dst_ptrs = _i64(t.data_ptr() for t in self.dst)
        self.src_ptrs = _i64(t.data_ptr() for t in self.src)
        self.feat = _i64(r * p for r, p in zip(self.rows, self.pitches))
        self.row_pitch = _i64(self.pitches)


def _launch(b: _Banks, dst_idx, src_idx, copy_bytes=None, num=None, bpb=8):
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

    fast_index_copy_multi_jit(
        b.dst_ptrs, b.src_ptrs, b.feat, dst_idx, src_idx, num,
        copy_bytes=None if copy_bytes is None else _i64(copy_bytes),
        row_pitch=None if copy_bytes is None else b.row_pitch,
        blocks_per_bank=bpb,
    )
    torch.cuda.synchronize()


def _indices(n: int, seed: int = 7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    src = torch.randperm(NUM_EXPERTS, generator=g, device="cuda")[:n].to(torch.int32)
    dst = torch.randperm(SLOTS, generator=g, device="cuda")[:n].to(torch.int32)
    return dst.contiguous(), src.contiguous()


def _check(b: _Banks, dst_idx, src_idx, widths, live=None):
    """Every addressed slot holds the source's ``widths[bank]`` prefix of each weight
    row, and every other byte of the cache is still poison."""
    di, si = dst_idx.long(), src_idx.long()
    if live is not None:
        di, si = di[:live], si[:live]
    poison = torch.full((1,), POISON, dtype=torch.uint8, device="cuda")
    for src, cache, w in zip(b.src, b.dst, widths):
        got, want = cache[di], src[si]
        assert torch.equal(got[:, :, :w], want[:, :, :w]), "payload differs"
        assert (got[:, :, w:] == poison).all(), (
            "the destination tail was written -- either the kernel is still moving "
            "those bytes, or it treated the payload as a prefix of the whole expert"
        )
        touched = torch.zeros(b.slots, dtype=torch.bool, device="cuda")
        touched[di] = True
        assert (cache[~touched] == poison).all(), "an unaddressed slot was written"


# --------------------------------------------------------------------------- #
# 1. the default path is untouched
# --------------------------------------------------------------------------- #


def test_absent_copy_bytes_is_the_full_pitch():
    n = 12
    dst_idx, src_idx = _indices(n)

    ref = _Banks(seed=1)
    _launch(ref, dst_idx, src_idx)

    same = _Banks(seed=1)
    _launch(same, dst_idx, src_idx, copy_bytes=[GU_PITCH, DN_PITCH])

    for a, c in zip(ref.dst, same.dst):
        assert torch.equal(a, c), "copy_bytes == row_pitch diverged from the default path"
    # ...and the default path really did move everything.
    for src, cache in zip(ref.src, ref.dst):
        assert torch.equal(cache[dst_idx.long()], src[src_idx.long()])


def test_env_kill_switch_forces_the_full_pitch(monkeypatch):
    from freetoken.kernel.fast_index_copy import UNPITCHED_COPY_ENV

    dst_idx, src_idx = _indices(8, seed=9)
    monkeypatch.setenv(UNPITCHED_COPY_ENV, "1")
    b = _Banks(seed=2)
    _launch(b, dst_idx, src_idx, copy_bytes=[GU_NATIVE, DN_NATIVE])
    for src, cache in zip(b.src, b.dst):
        assert torch.equal(cache[dst_idx.long()], src[src_idx.long()]), (
            f"{UNPITCHED_COPY_ENV}=1 did not restore the full-pitch copy"
        )
    monkeypatch.delenv(UNPITCHED_COPY_ENV)
    assert os.getenv(UNPITCHED_COPY_ENV) is None


def test_copy_bytes_and_row_pitch_must_agree():
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

    dst_idx, src_idx = _indices(4, seed=10)
    b = _Banks(seed=2)
    with pytest.raises(AssertionError, match="given together"):
        fast_index_copy_multi_jit(
            b.dst_ptrs, b.src_ptrs, b.feat, dst_idx, src_idx, None,
            copy_bytes=_i64([GU_NATIVE, DN_NATIVE]),
        )


# --------------------------------------------------------------------------- #
# 2. narrow payload: exact prefix per weight row, untouched tails
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "widths",
    [
        # The shipped case: both banks at their native packed widths.
        (GU_NATIVE, DN_NATIVE),
        # One bank narrowed, one at full pitch -- an IQ2_XS gate_up layer alongside
        # an MXFP4 down layer, which is exactly layers 26 and 42.
        (GU_NATIVE, DN_PITCH),
        (GU_PITCH, DN_NATIVE),
        # Minimum: a single 16 B unit per weight row.
        (16, 16),
    ],
)
def test_prefix_exact_and_tail_preserved(widths):
    dst_idx, src_idx = _indices(10, seed=11)
    b = _Banks(seed=3)
    _launch(b, dst_idx, src_idx, copy_bytes=list(widths))
    _check(b, dst_idx, src_idx, widths)


def test_source_tail_is_never_read():
    """Two banks agreeing on every weight row's payload and disagreeing in every tail
    must give the same result -- the read-side mirror of the tail test."""
    dst_idx, src_idx = _indices(9, seed=13)
    widths = (GU_NATIVE, DN_NATIVE)

    a = _Banks(seed=4)
    c = _Banks(seed=4)
    for src, w in zip(c.src, widths):
        src[:, :, w:] = 0x5A
    for sa, sc, w in zip(a.src, c.src, widths):
        assert torch.equal(sa[:, :, :w], sc[:, :, :w])
        assert not torch.equal(sa[:, :, w:], sc[:, :, w:])

    _launch(a, dst_idx, src_idx, copy_bytes=list(widths))
    _launch(c, dst_idx, src_idx, copy_bytes=list(widths))
    for x, y in zip(a.dst, c.dst):
        assert torch.equal(x, y)


def test_valid_length_still_bounds_the_rows():
    """``num_indices`` truncation and payload narrowing are independent axes."""
    n, live = 12, 5
    dst_idx, src_idx = _indices(n, seed=17)
    num = torch.tensor([live], dtype=torch.int64, device="cuda")
    b = _Banks(seed=5)
    widths = (GU_NATIVE, DN_NATIVE)
    _launch(b, dst_idx, src_idx, copy_bytes=list(widths), num=num)
    _check(b, dst_idx, src_idx, widths, live=live)


@pytest.mark.parametrize("bpb", [1, 8, 64])
def test_grid_width_does_not_change_the_result(bpb):
    """The three blocks_per_bank the cache actually uses (prefetch, serial, gather):
    the grid-stride bounds are recomputed from the narrowed unit count, so an off-by-one
    there would show up only at some widths."""
    dst_idx, src_idx = _indices(11, seed=19)
    widths = (GU_NATIVE, DN_NATIVE)
    b = _Banks(seed=6)
    _launch(b, dst_idx, src_idx, copy_bytes=list(widths), bpb=bpb)
    _check(b, dst_idx, src_idx, widths)


# --------------------------------------------------------------------------- #
# 3. the point of the exercise: fewer bytes on the wire
# --------------------------------------------------------------------------- #


def _bench_one_bank(pitch: int, widths, experts=128, rows=1024, iters=30):
    """Time a single pinned-host bank at ``pitch`` for each payload width.

    Returns ``{width_or_None: (ms, effective GB/s)}``. ``None`` is the unpitched
    kernel. Enqueue-only inside the timed region: a per-iteration synchronize would
    time the launch round trip instead of the transfer.
    """
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.kernel.pinned import device_ptr

    src = torch.empty((experts, rows, pitch), dtype=torch.uint8).pin_memory()
    dst = torch.empty((experts, rows, pitch), dtype=torch.uint8, device="cuda")
    dst_ptrs, src_ptrs = _i64([dst.data_ptr()]), _i64([device_ptr(src)])
    feat, rp = _i64([rows * pitch]), _i64([pitch])
    idx = torch.arange(experts, dtype=torch.int32, device="cuda")

    out = {}
    for w in widths:
        cb = None if w is None else _i64([w])
        r = None if w is None else rp

        def one():
            fast_index_copy_multi_jit(
                dst_ptrs, src_ptrs, feat, idx, idx, None,
                copy_bytes=cb, row_pitch=r, blocks_per_bank=8,
            )

        for _ in range(5):
            one()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            one()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        moved = experts * rows * (pitch if w is None else w)
        out[w] = (ms, moved / ms / 1e6)
    del src, dst
    torch.cuda.empty_cache()
    return out


def test_gate_up_geometry_saves_wall_time():
    """The shipped win: 1184 B IQ2_XS rows in a 1568 B bank.

    Fewer bytes is not automatically less time -- the source is pinned host memory read
    over PCIe, and skipping a sub-page gap in every row costs read efficiency. At this
    geometry it pays: 75.5% of the bytes at ~95% of the full-pitch bandwidth. The
    assertion is on WALL TIME, because that is the only thing the decode loop feels.
    """
    r = _bench_one_bank(GU_PITCH, (None, GU_PITCH, GU_NATIVE))
    full_ms, full_gbs = r[None]
    same_ms, _ = r[GU_PITCH]
    nat_ms, nat_gbs = r[GU_NATIVE]
    print(
        f"\n  gate_up pitch {GU_PITCH}: unpitched {full_ms:.3f} ms ({full_gbs:.1f} GB/s) | "
        f"pitched@{GU_PITCH} {same_ms:.3f} ms | "
        f"pitched@{GU_NATIVE} {nat_ms:.3f} ms ({nat_gbs:.1f} GB/s), "
        f"{100 * nat_ms / full_ms:.0f}% of the time for "
        f"{100 * GU_NATIVE / GU_PITCH:.0f}% of the bytes"
    )
    # The pitched kernel must cost nothing when it is not narrowing anything: this is
    # what justifies leaving the specialization enabled for a mixed bank.
    assert same_ms < full_ms * 1.05, (
        f"the pitched kernel at full width costs {same_ms / full_ms:.2f}x -- the "
        "if-constexpr specialization is not doing its job"
    )
    assert nat_ms < full_ms * 0.92, (
        f"narrowing gate_up rows to {GU_NATIVE} B only gave {full_ms / nat_ms:.2f}x; "
        "expected ~1.25x. The pitched path is not reducing PCIe time."
    )


def test_down_geometry_is_the_documented_exception():
    """1088 -> 784 is the case where narrowing LOSES, and it is measured, not assumed.

    Reading 784 of every 1088 B moves 72% of the bytes but takes ~150% of the wall
    time on this box (PCIe 5.0 under WSL2 GPU-PV): the read stream stops being one
    contiguous run and the request count, not the byte count, becomes the limit. The
    same skip in a 1568 B row is fine (test above), so this is a property of the
    geometry rather than of the kernel -- which is exactly why the loader declares
    payload widths per BANK and this one keeps its full pitch.

    If a future box or driver makes this profitable, this test fails and the policy in
    ``gguf_experts.q2k_ud_layer_copy_bytes`` should be revisited. It is written to fail
    on good news, deliberately.
    """
    r = _bench_one_bank(DN_PITCH, (None, DN_NATIVE))
    full_ms, full_gbs = r[None]
    nat_ms, nat_gbs = r[DN_NATIVE]
    print(
        f"\n  down pitch {DN_PITCH}: unpitched {full_ms:.3f} ms ({full_gbs:.1f} GB/s) | "
        f"pitched@{DN_NATIVE} {nat_ms:.3f} ms ({nat_gbs:.1f} GB/s), "
        f"{100 * nat_ms / full_ms:.0f}% of the time for "
        f"{100 * DN_NATIVE / DN_PITCH:.0f}% of the bytes"
    )
    assert nat_ms > full_ms, (
        f"narrowing down rows to {DN_NATIVE} B is now FASTER "
        f"({full_ms / nat_ms:.2f}x) -- good news, but it invalidates the policy that "
        "keeps the down bank on full-pitch copies. Re-measure and update "
        "q2k_ud_layer_copy_bytes."
    )
