"""CPU-only tests for the FREETOKEN_DUMMY_VRAM_MIB measurement switch.

This switch exists to separate "memory taken" from "vision tower built" (see
Engine.__init__ around the model-weights-loaded point in engine.py). It must be an
exact no-op when unset, must compute the right byte count when set, and must never
silently disable itself on a malformed value -- a measurement instrument that goes
quiet instead of erroring would corrupt the experiment it exists to run.
"""

import pytest

from freetoken.engine.engine import _allocate_dummy_vram, _dummy_vram_bytes_from_env

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
