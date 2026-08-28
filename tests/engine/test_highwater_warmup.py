"""CPU-only sizing tests for allocator high-water warmup."""

from freetoken.engine.engine import _highwater_prefill_len


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
