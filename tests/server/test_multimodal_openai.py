"""API-layer coverage for carrying images from the OpenAI HTTP API into a GenSpec /
TokenizeMsg: image_url content-part parsing (data: URI, local path, http(s) gate), the
`_flatten_text_parts` fix (text-only pinned, image parts no longer raise), and the
full `handle_chat_completion` wiring (state.sent.images / state.sent.text).

No GPU, no server boot: everything here exercises the request -> GenSpec -> TokenizeMsg
conversion with a FakeState, exactly like tests/server/test_openai_api.py does.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server import openai_api
from freetoken.server.api_models import ChatCompletionRequest, Message, MessageContent
from freetoken.server.generation import _flatten_text_parts
from freetoken.server.openai_api import chat_request_to_genspec, handle_chat_completion


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------------------
# _flatten_text_parts: pin the existing text-only behavior, confirm images no longer raise
# --------------------------------------------------------------------------------------


def test_flatten_text_parts_pinned_all_text_behavior():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _flatten_text_parts(parts) == "ab"


def test_flatten_text_parts_missing_text_key_defaults_to_empty_string_pinned():
    parts = [{"type": "text"}, {"type": "text", "text": None}]
    assert _flatten_text_parts(parts) == ""


def test_flatten_text_parts_still_raises_on_a_genuinely_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported content part type"):
        _flatten_text_parts([{"type": "audio_url"}])


def test_flatten_text_parts_image_parts_no_longer_raise_and_order_is_preserved():
    parts = [
        {"type": "text", "text": "look: "},
        {"type": "image_url", "image_url": True},
        {"type": "text", "text": " and "},
        {"type": "image_url", "image_url": True},
        {"type": "text", "text": " together"},
    ]
    # Must not raise, and must NOT be flattened to a string (the template needs the
    # per-item structure to render each marker in place).
    assert _flatten_text_parts(parts) == parts


# --------------------------------------------------------------------------------------
# image_url decoding: data: URI, local path, http(s) gate, malformed input
# --------------------------------------------------------------------------------------


def test_data_uri_image_decoded_to_exact_bytes():
    raw = bytes(range(256))
    b64 = base64.b64encode(raw).decode()
    got = openai_api._decode_image_url(
        {"url": f"data:image/png;base64,{b64}"}, allow_remote_images=False
    )
    assert got == raw


def test_data_uri_as_bare_string_value_also_works():
    raw = b"bare-string-form"
    b64 = base64.b64encode(raw).decode()
    got = openai_api._decode_image_url(f"data:image/jpeg;base64,{b64}", allow_remote_images=False)
    assert got == raw


def test_malformed_data_uri_missing_comma_gives_clean_error():
    with pytest.raises(ValueError, match="malformed data URI"):
        openai_api._decode_image_url("data:image/png;base64", allow_remote_images=False)


def test_malformed_data_uri_not_base64_gives_clean_error():
    with pytest.raises(ValueError, match="malformed data URI"):
        openai_api._decode_image_url("data:image/png,plain-text-not-base64", allow_remote_images=False)


def test_malformed_base64_payload_gives_clean_error():
    with pytest.raises(ValueError, match="malformed base64"):
        openai_api._decode_image_url(
            "data:image/png;base64,not-valid-base64!!!", allow_remote_images=False
        )


def test_local_path_image_is_read(tmp_path):
    p = tmp_path / "img.bin"
    p.write_bytes(b"local-filesystem-image-bytes")
    got = openai_api._decode_image_url(str(p), allow_remote_images=False)
    assert got == b"local-filesystem-image-bytes"


def test_missing_local_path_gives_clean_error_not_a_traceback():
    with pytest.raises(ValueError, match="could not read local image path"):
        openai_api._decode_image_url("/no/such/file/anywhere.png", allow_remote_images=False)


def test_remote_url_rejected_by_default():
    with pytest.raises(ValueError, match="remote image URLs are disabled"):
        openai_api._decode_image_url(
            "https://example.com/cat.png", allow_remote_images=False
        )


def test_remote_url_fetched_when_opted_in(monkeypatch):
    calls = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"fetched-remote-bytes"

    def fake_urlopen(url, timeout=10):
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    got = openai_api._decode_image_url(
        "https://example.com/cat.png", allow_remote_images=True
    )
    assert got == b"fetched-remote-bytes"
    assert calls == ["https://example.com/cat.png"]


# --------------------------------------------------------------------------------------
# _messages_and_images / chat_request_to_genspec: end-to-end structure + scrubbing
# --------------------------------------------------------------------------------------


def test_messages_and_images_two_images_mixed_with_text_in_order():
    raw0 = b"first-image-bytes"
    raw1 = b"second-image-bytes"
    b64_0 = base64.b64encode(raw0).decode()
    b64_1 = base64.b64encode(raw1).decode()
    msgs = [
        Message(
            role="user",
            content=[
                MessageContent(type="text", text="compare these:"),
                MessageContent(type="image_url", image_url={"url": f"data:image/png;base64,{b64_0}"}),
                MessageContent(type="text", text="vs"),
                MessageContent(type="image_url", image_url={"url": f"data:image/png;base64,{b64_1}"}),
            ],
        )
    ]
    dumped, images = openai_api._messages_and_images(msgs, allow_remote_images=False)
    assert images == [raw0, raw1]
    # The heavy base64 payload must be scrubbed -- only the key's presence matters to
    # the chat template (chat_template.jinja: `'image_url' in item`).
    assert dumped[0]["content"] == [
        {"type": "text", "text": "compare these:"},
        {"type": "image_url", "image_url": True},
        {"type": "text", "text": "vs"},
        {"type": "image_url", "image_url": True},
    ]
    for part in dumped[0]["content"]:
        if part["type"] == "image_url":
            assert part["image_url"] is not None
            assert "base64" not in str(part["image_url"])


def test_chat_request_to_genspec_text_only_is_pinned_to_a_flat_string():
    req = ChatCompletionRequest(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": " there"}],
            }
        ],
    )
    spec = chat_request_to_genspec(req, {})
    assert spec.images is None
    assert spec.messages[0]["content"] == "hi there"


def test_chat_request_to_genspec_plain_string_content_is_untouched():
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hello"}])
    spec = chat_request_to_genspec(req, {})
    assert spec.images is None
    assert spec.messages[0]["content"] == "hello"


def test_chat_request_to_genspec_rejects_remote_image_by_default():
    req = ChatCompletionRequest(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://evil.example/x.png"}}],
            }
        ],
    )
    with pytest.raises(ValueError, match="remote image URLs are disabled"):
        chat_request_to_genspec(req, {})


# --------------------------------------------------------------------------------------
# Full handle_chat_completion wiring: state.sent (the TokenizeMsg) carries the images
# --------------------------------------------------------------------------------------


class FakeState:
    def __init__(self, replies: list[UserReply], allow_remote_images: bool | None = None) -> None:
        config_kwargs = dict(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=None,
        )
        if allow_remote_images is not None:
            config_kwargs["allow_remote_images"] = allow_remote_images
        self.config = SimpleNamespace(**config_kwargs)
        self.replies = replies
        self.sent: TokenizeMsg | None = None

    def new_user(self) -> int:
        return 77

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 77
        for reply in self.replies:
            yield reply


def _image_chat_request() -> tuple[ChatCompletionRequest, bytes]:
    raw = b"end-to-end-image-bytes"
    b64 = base64.b64encode(raw).decode()
    req = ChatCompletionRequest(
        model="client-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=8,
    )
    return req, raw


def test_handle_chat_completion_sends_images_on_the_tokenize_msg():
    req, raw = _image_chat_request()
    state = FakeState([UserReply(uid=77, incremental_output="ok", finished=True)])

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    assert state.sent is not None
    assert state.sent.images == [raw]
    assert state.sent.text[0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": True},
    ]
    assert response["choices"][0]["message"]["content"] == "ok"


def test_handle_chat_completion_defaults_to_rejecting_remote_images_when_flag_absent():
    """A FakeState whose config predates this field (no allow_remote_images attribute,
    exactly like the existing fixture in test_openai_api.py) must not crash -- it must
    behave as allow_remote_images=False, not raise AttributeError."""
    req = ChatCompletionRequest(
        model="client-model",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}],
            }
        ],
    )
    state = FakeState([])  # config has no allow_remote_images at all

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    assert state.sent is None  # request was rejected before submission
    assert response.status_code == 400
    body = json.loads(response.body)
    assert "remote image URLs are disabled" in body["error"]["message"]


def test_handle_chat_completion_text_only_unaffected_by_images_plumbing():
    """Regression pin: a text-only request must still send TokenizeMsg.images == None
    and a flattened string content, exactly like before this feature existed."""
    state = FakeState([UserReply(uid=77, incremental_output="hi", finished=True)])
    req = ChatCompletionRequest(
        model="client-model", messages=[{"role": "user", "content": "hello"}], max_tokens=4,
    )

    run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    assert state.sent is not None
    assert state.sent.images is None
    assert state.sent.text == [{"role": "user", "content": "hello"}]
