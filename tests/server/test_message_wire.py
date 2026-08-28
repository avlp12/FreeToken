"""Encoder/decoder round-trips for the ZMQ control messages (no GPU).

Every message that crosses api -> tokenizer -> scheduler -> tokenizer -> api must survive the
wire with its fields intact; these pin the ones carrying state a later consumer reads back
(rebuild control, prompt admission, per-reply token deltas and KV usage).
"""

from __future__ import annotations

import msgpack

from freetoken.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.core import SamplingParams


def _wire_round_trip(base_cls, msg):
    """The actual wire path a message takes (ZmqPushQueue.put / ZmqPullQueue.get, see
    freetoken/utils/mp.py): encoder -> msgpack.packb(..., use_bin_type=True) ->
    msgpack.unpackb(..., raw=False) -> decoder. Exercises the real bytes-over-the-wire
    behavior, not just the dict-level encoder/decoder step the other tests above use."""
    packed = msgpack.packb(base_cls.encoder(msg), use_bin_type=True)
    unpacked = msgpack.unpackb(packed, raw=False)
    return base_cls.decoder(unpacked)


def test_cache_rebuild_msg_roundtrip():
    msg = CacheRebuildMsg(request_id="abc", moe_cache_size=8, num_pages=1024, mode="if_idle")
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("abc", 8, 1024, "if_idle")


def test_cache_rebuild_backend_msg_roundtrip():
    msg = CacheRebuildBackendMsg(request_id="r1", moe_cache_size=None, num_pages=256, mode="drain")
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, CacheRebuildBackendMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("r1", None, 256, "drain")


def test_cache_rebuild_result_msg_roundtrip():
    msg = CacheRebuildResultMsg(request_id="r2", status="ok", moe_cache_size=16, num_pages=512)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildResultMsg)
    assert (out.request_id, out.status, out.moe_cache_size, out.num_pages, out.error) == (
        "r2", "ok", 16, 512, None,
    )


def test_cache_rebuild_reply_roundtrip():
    msg = CacheRebuildReply(request_id="r3", status="failed", error="boom")
    out = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))
    assert isinstance(out, CacheRebuildReply)
    assert (out.request_id, out.status, out.error) == ("r3", "failed", "boom")


def test_prompt_admitted_msg_roundtrip():
    msg = PromptAdmittedMsg(uid=42, prompt_tokens=1234, cached_tokens=500)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, PromptAdmittedMsg)
    assert (out.uid, out.prompt_tokens, out.cached_tokens) == (42, 1234, 500)


def test_user_reply_token_deltas_round_trip():
    msg = UserReply(
        uid=7,
        incremental_output="hello",
        finished=False,
        prompt_tokens_delta=11,
        completion_tokens_delta=3,
        cached_tokens=4,
        kv_used_pages=40,
        kv_total_pages=512,
        gpu_mem_bytes=64 * (1 << 30),
    )

    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))

    assert isinstance(decoded, UserReply)
    assert decoded.uid == 7
    assert decoded.incremental_output == "hello"
    assert decoded.finished is False
    assert decoded.prompt_tokens_delta == 11
    assert decoded.completion_tokens_delta == 3
    assert decoded.cached_tokens == 4
    assert decoded.kv_used_pages == 40
    assert decoded.kv_total_pages == 512
    assert decoded.gpu_mem_bytes == 64 * (1 << 30)


def test_detokenize_msg_carries_kv_usage_round_trip():
    msg = DetokenizeMsg(
        uid=3, next_token=42, finished=True, token_ids=[40, 41, 42],
        kv_used_pages=10, kv_total_pages=256, gpu_mem_bytes=1 << 30,
        mamba_used_slots=7, mamba_total_slots=64,
        swa_used_tokens=8448, swa_total_tokens=76800,
    )
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert decoded.token_ids == [40, 41, 42]
    assert (decoded.kv_used_pages, decoded.kv_total_pages, decoded.gpu_mem_bytes) == (10, 256, 1 << 30)
    assert (decoded.mamba_used_slots, decoded.mamba_total_slots) == (7, 64)
    assert (decoded.swa_used_tokens, decoded.swa_total_tokens) == (8448, 76800)


def test_tokenize_msg_images_round_trip_through_msgpack():
    """Raw image bytes must cross api -> tokenizer unmodified: bytes serialize natively
    (unlike a 2D pixel tensor, which `serialize_type` refuses -- see message/utils.py),
    so this exercises the real msgpack wire, not just serialize_type/deserialize_type."""
    raw0 = bytes(range(256)) * 4  # exercise every byte value, including NUL and 0xff
    raw1 = b""  # a zero-length image part must not be conflated with "no image"
    msg = TokenizeMsg(
        uid=5,
        text=[{"role": "user", "content": [{"type": "image_url", "image_url": True}]}],
        sampling_params=SamplingParams(max_tokens=16),
        images=[raw0, raw1],
    )
    out = _wire_round_trip(BaseTokenizerMsg, msg)
    assert isinstance(out, TokenizeMsg)
    assert out.images == [raw0, raw1]
    assert isinstance(out.images[0], bytes) and isinstance(out.images[1], bytes)
    assert out.text == msg.text


def test_tokenize_msg_text_only_images_field_round_trips_as_none():
    """The text-only path (images unset) must stay exactly None across the wire --
    pinning that a request with no images takes the pre-existing code path unchanged."""
    msg = TokenizeMsg(uid=6, text="hello", sampling_params=SamplingParams())
    out = _wire_round_trip(BaseTokenizerMsg, msg)
    assert isinstance(out, TokenizeMsg)
    assert out.images is None


def test_user_msg_images_round_trip_through_msgpack():
    """images carried on UserMsg (tokenizer -> scheduler) must also survive the real
    wire byte-identical, in order, so the core process preprocesses the right bytes for
    the right placeholder run."""
    import torch

    raw0 = b"\x00\x01\xffPNG-ish-bytes"
    raw1 = b"second-image-entirely-different-bytes"
    msg = UserMsg(
        uid=9,
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[raw0, raw1],
    )
    out = _wire_round_trip(BaseBackendMsg, msg)
    assert isinstance(out, UserMsg)
    assert out.images == [raw0, raw1]
    assert out.input_ids.tolist() == [1, 2, 3]
    assert out.mm_embeds is None


def test_user_msg_text_only_images_field_round_trips_as_none():
    import torch

    msg = UserMsg(
        uid=10, input_ids=torch.tensor([1, 2], dtype=torch.int32), sampling_params=SamplingParams()
    )
    out = _wire_round_trip(BaseBackendMsg, msg)
    assert isinstance(out, UserMsg)
    assert out.images is None


def test_client_dicts_with_the_wire_tag_key_survive_intact():
    """Tool JSON Schemas and chat_template_kwargs are free-form client data. A field literally
    named ``__type__`` (a common discriminator) must not be read back as a serialized class --
    that used to kill the tokenizer worker on an unknown/incompatible name."""
    hostile = [
        {"__type__": "AbortMsg"},                                    # a real class name
        {"__type__": "NoSuchClassAnywhere"},                         # an unknown one
        {"type": "object", "properties": {"__type__": {"type": "string"}}},
        {"__raw_dict__": {"a": 1}},                                  # collides with the escape key
        {"deep": {"__type__": "AbortMsg", "l": [{"__type__": "x"}]}},
    ]
    for payload in hostile:
        msg = TokenizeMsg(
            uid=1, text="hi", sampling_params=SamplingParams(),
            chat_template_kwargs=payload,
            tools=[{"type": "function", "function": {"name": "f", "parameters": payload}}],
        )
        out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
        assert isinstance(out, TokenizeMsg)
        assert out.chat_template_kwargs == payload
        assert out.tools[0]["function"]["parameters"] == payload
