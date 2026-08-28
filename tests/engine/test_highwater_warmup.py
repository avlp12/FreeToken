"""CPU-only sizing tests for allocator high-water warmup."""

from freetoken.engine.engine import (
    _VRAM_HEADROOM_WARN_THRESHOLD_BYTES,
    _highwater_prefill_len,
    _vram_headroom_warning,
)


def test_default_launcher_chunk():
    assert _highwater_prefill_len(65536, 8192, 8192, 65536) == 8192


def test_longctx_still_chunk_not_context():
    assert _highwater_prefill_len(250000, 8192, 8192, 250000) == 8192


def test_plain_engineconfig_does_not_warm_full_context():
    # EngineConfig.max_forward_len == max_seq_len; must not request 65k scratch.
    assert _highwater_prefill_len(65536, 65536, None, 65536) == 8192


def test_tiny_seq_is_noop():
    assert _highwater_prefill_len(1, 8192, 8192, 8192) is None


def test_page_table_caps_length():
    assert _highwater_prefill_len(65536, 8192, 8192, 4096) == 4096


def test_explicit_16384_chunk():
    assert _highwater_prefill_len(65536, 16384, 16384, 65536) == 16384


def test_headroom_comfortably_above_threshold_is_silent():
    # 0.5 GiB free, well clear of the 0.1 GiB band.
    assert _vram_headroom_warning(int(0.5 * 1024**3)) is None


def test_headroom_below_threshold_warns_with_value_threshold_and_remedy():
    free_bytes = int(0.03 * 1024**3)
    msg = _vram_headroom_warning(free_bytes)
    assert msg is not None
    assert "0.03 GiB" in msg
    assert "0.10 GiB" in msg
    assert "moe-cache-size" in msg


def test_headroom_exactly_at_threshold_is_not_warned():
    # Boundary choice: "below" the threshold is exclusive, so free VRAM equal
    # to the threshold itself is still considered safe and does not warn.
    assert _vram_headroom_warning(_VRAM_HEADROOM_WARN_THRESHOLD_BYTES) is None


def test_zero_free_bytes_warns():
    msg = _vram_headroom_warning(0)
    assert msg is not None
    assert "0.00 GiB" in msg
    assert "moe-cache-size" in msg
