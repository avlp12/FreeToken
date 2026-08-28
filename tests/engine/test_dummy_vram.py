"""CPU-only tests for the FREETOKEN_DUMMY_VRAM_MIB measurement switch.

This switch exists to separate "memory taken" from "vision tower built" (see
Engine.__init__ around the model-weights-loaded point in engine.py). It must be an
exact no-op when unset, must compute the right byte count when set, and must never
silently disable itself on a malformed value -- a measurement instrument that goes
quiet instead of erroring would corrupt the experiment it exists to run.

FREETOKEN_DUMMY_VRAM_CHUNKS is a companion switch: it splits the same total byte
count into N separate device tensors instead of one, to test whether the vision
tower's prefill cost is granularity (333 separate parameter tensors) rather than
occupancy -- a single-block dummy of the same size measured no cost at all. Same
no-op and no-silent-failure requirements apply.
"""

import pytest

from freetoken.engine.engine import (
    _allocate_dummy_vram,
    _allocate_dummy_vram_chunks,
    _dummy_vram_bytes_from_env,
    _dummy_vram_chunk_sizes,
    _dummy_vram_chunks_from_env,
)

MIB = 1024 * 1024


def test_unset_is_noop(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DUMMY_VRAM_MIB", raising=False)
    assert _dummy_vram_bytes_from_env() == 0


def test_zero_is_noop(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "0")
    assert _dummy_vram_bytes_from_env() == 0


def test_unset_allocator_makes_no_cuda_call(monkeypatch):
    # A CPU device is fine here: unset must short-circuit to None before the
    # function ever touches torch.cuda / does a device allocation. If it didn't
    # short-circuit, torch.empty(..., device="cpu") would still succeed (it's not
    # actually a CUDA call) but would return a real tensor instead of None, so
    # this also pins down the no-op *result*, not just the absence of a crash.
    monkeypatch.delenv("FREETOKEN_DUMMY_VRAM_MIB", raising=False)
    assert _allocate_dummy_vram("cpu") is None


def test_zero_allocator_makes_no_cuda_call(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "0")
    assert _allocate_dummy_vram("cpu") is None


@pytest.mark.parametrize(
    "mib, expected_bytes",
    [
        ("1", 1 * MIB),
        ("100", 100 * MIB),
        ("850", 850 * MIB),  # ~0.83 GiB, the vision tower's measured footprint
        ("3000", 3000 * MIB),
    ],
)
def test_byte_count_conversion(monkeypatch, mib, expected_bytes):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", mib)
    assert _dummy_vram_bytes_from_env() == expected_bytes


def test_whitespace_is_tolerated(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "  512  ")
    assert _dummy_vram_bytes_from_env() == 512 * MIB


@pytest.mark.parametrize("bad_value", ["abc", "1.5", "3000MiB", "0x10", ""])
def test_malformed_value_raises(monkeypatch, bad_value):
    if bad_value == "":
        # Empty string is treated the same as unset/0 -- deliberately a no-op,
        # not an error, so it doesn't trip a shell `FOO=` default-empty pattern.
        monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", bad_value)
        assert _dummy_vram_bytes_from_env() == 0
        return
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", bad_value)
    with pytest.raises(ValueError, match="FREETOKEN_DUMMY_VRAM_MIB"):
        _dummy_vram_bytes_from_env()


@pytest.mark.parametrize("bad_value", ["-1", "-100"])
def test_negative_value_raises(monkeypatch, bad_value):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", bad_value)
    with pytest.raises(ValueError, match="must not be negative"):
        _dummy_vram_bytes_from_env()


# ------------------------- FREETOKEN_DUMMY_VRAM_CHUNKS -------------------------


def test_chunks_unset_defaults_to_one(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DUMMY_VRAM_CHUNKS", raising=False)
    assert _dummy_vram_chunks_from_env() == 1


def test_chunks_empty_defaults_to_one(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", "")
    assert _dummy_vram_chunks_from_env() == 1


def test_chunks_explicit_one(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", "1")
    assert _dummy_vram_chunks_from_env() == 1


@pytest.mark.parametrize("bad_value", ["abc", "1.5", "3MiB", "0x10"])
def test_chunks_malformed_value_raises(monkeypatch, bad_value):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", bad_value)
    with pytest.raises(ValueError, match="FREETOKEN_DUMMY_VRAM_CHUNKS"):
        _dummy_vram_chunks_from_env()


@pytest.mark.parametrize("bad_value", ["0", "-5", "-1"])
def test_chunks_below_one_raises(monkeypatch, bad_value):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", bad_value)
    with pytest.raises(ValueError, match="FREETOKEN_DUMMY_VRAM_CHUNKS"):
        _dummy_vram_chunks_from_env()


# ---------------------- _dummy_vram_chunk_sizes (pure, no GPU) ----------------------


def test_chunk_sizes_single_chunk_matches_total():
    # N=1 must be a "single-chunk plan" identical in total to today's one-tensor
    # behaviour: the whole byte count in one element.
    total = 856 * MIB
    assert _dummy_vram_chunk_sizes(total, 1) == [total]


def test_chunk_sizes_even_split():
    total = 100 * MIB
    sizes = _dummy_vram_chunk_sizes(total, 10)
    assert sizes == [10 * MIB] * 10
    assert sum(sizes) == total


def test_chunk_sizes_856mib_over_333_sums_exactly_and_no_zero_chunk():
    # The real case this switch was built to test: the vision tower is 333
    # separate parameter tensors and its measured footprint is ~856 MiB.
    total = 856 * MIB
    n_chunks = 333
    sizes = _dummy_vram_chunk_sizes(total, n_chunks)
    assert len(sizes) == n_chunks
    assert sum(sizes) == total
    assert all(size > 0 for size in sizes)
    # Remainder handling: sizes differ by at most 1 byte, first `remainder`
    # chunks carry the extra byte.
    base, remainder = divmod(total, n_chunks)
    assert sizes[:remainder] == [base + 1] * remainder
    assert sizes[remainder:] == [base] * (n_chunks - remainder)


@pytest.mark.parametrize("n_chunks", [0, -1, -5])
def test_chunk_sizes_rejects_non_positive_n_chunks(n_chunks):
    with pytest.raises(ValueError, match="n_chunks"):
        _dummy_vram_chunk_sizes(100 * MIB, n_chunks)


# --------------------- _allocate_dummy_vram_chunks (CPU, no CUDA) ---------------------


def test_allocator_chunks_unset_mib_is_noop_regardless_of_chunks(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DUMMY_VRAM_MIB", raising=False)
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", "333")
    assert _allocate_dummy_vram_chunks("cpu") == []


def test_allocator_chunks_zero_mib_is_noop_regardless_of_chunks(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "0")
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", "333")
    assert _allocate_dummy_vram_chunks("cpu") == []


@pytest.mark.parametrize("bad_chunks", ["abc", "1.5", "0", "-5"])
def test_allocator_chunks_malformed_value_is_still_noop_when_mib_unset(
    monkeypatch, bad_chunks
):
    # FREETOKEN_DUMMY_VRAM_CHUNKS is not even read when the MIB switch is off --
    # a malformed chunk count must not corrupt an experiment that isn't running.
    monkeypatch.delenv("FREETOKEN_DUMMY_VRAM_MIB", raising=False)
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", bad_chunks)
    assert _allocate_dummy_vram_chunks("cpu") == []


def test_allocator_chunks_unset_is_single_chunk_matching_legacy_allocator(
    monkeypatch,
):
    # Unset FREETOKEN_DUMMY_VRAM_CHUNKS must delegate to the pre-chunking
    # _allocate_dummy_vram -- provably identical, not just equivalent -- so
    # this compares the two allocators' outputs directly.
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "8")
    monkeypatch.delenv("FREETOKEN_DUMMY_VRAM_CHUNKS", raising=False)
    chunks = _allocate_dummy_vram_chunks("cpu")
    legacy = _allocate_dummy_vram("cpu")
    assert len(chunks) == 1
    assert chunks[0].shape == legacy.shape
    assert chunks[0].dtype == legacy.dtype
    assert chunks[0].numel() * chunks[0].element_size() == 8 * MIB


def test_allocator_chunks_explicit_one_is_single_chunk(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "8")
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", "1")
    chunks = _allocate_dummy_vram_chunks("cpu")
    assert len(chunks) == 1
    assert chunks[0].numel() * chunks[0].element_size() == 8 * MIB


def test_allocator_chunks_splits_into_n_tensors_summing_to_total(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_MIB", "856")
    monkeypatch.setenv("FREETOKEN_DUMMY_VRAM_CHUNKS", "333")
    chunks = _allocate_dummy_vram_chunks("cpu")
    assert len(chunks) == 333
    assert all(t.numel() * t.element_size() > 0 for t in chunks)
    total = sum(t.numel() * t.element_size() for t in chunks)
    assert total == 856 * MIB
