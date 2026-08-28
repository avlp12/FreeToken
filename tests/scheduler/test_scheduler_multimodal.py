"""Core-side coverage for `Scheduler._fill_mm_embeds` / `_process_one_msg`'s images
branch: preprocess + encode_images() runs before a request is admitted, and both
"fail loudly" paths the assignment calls out (no vision support; placeholder-slot vs
embedding-row mismatch) produce a clean per-request ErrorReplyMsg instead of a crash
or a silently wrong answer.

Follows the existing style in tests/scheduler/test_cost_accounting_core.py:
Scheduler.__new__(Scheduler) plus a handful of attributes, no Engine, no GPU. Uses the
real Qwen3.8-Flash-Next-FP8 checkpoint's image preprocessing (freetoken.models.qwen4_exp.
image, CPU-only) so the "success" / "mismatch" fixtures use real patch-grid math rather
than hand-picked numbers; skipped when that checkpoint is not present.
"""

from __future__ import annotations

import io
import os
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from freetoken.core import SamplingParams
from freetoken.message import ErrorReplyMsg, UserMsg
from freetoken.scheduler.scheduler import Scheduler

MODEL_PATH = "/root/models/Qwen3.8-Flash-Next-FP8"
IMAGE_TOKEN_ID = 248056

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH), reason="requires the real Qwen3.8-Flash-Next-FP8 checkpoint on disk"
)


def _png_bytes(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _fake_scheduler(model) -> tuple[Scheduler, list, list]:
    sched = Scheduler.__new__(Scheduler)
    sched.engine = SimpleNamespace(max_seq_len=4096, model=model)
    sched.config = SimpleNamespace(
        model_path=MODEL_PATH,
        model_config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
    )
    sched.device = torch.device("cpu")
    added: list = []
    sent: list = []
    sched.prefill_manager = SimpleNamespace(add_one_req=added.append)
    sched.send_result = sent.extend
    return sched, added, sent


# --------------------------------------------------------------------------------------
# _fill_mm_embeds, called directly (unbound, matching test_qwen4_exp_image.py's style
# for _merge_multimodal)
# --------------------------------------------------------------------------------------


def test_fill_mm_embeds_errors_when_model_has_no_vision_tower():
    """The production default: `encode_images` exists on the class unconditionally
    (Qwen4ExpForCausalLM.encode_images, model.py), but `vision_tower` is only built when
    FREETOKEN_LOAD_VISION=1. This is the realistic "vision not enabled" shape -- must be
    a clean per-request error, not an AttributeError reaching into self.vision_tower."""

    class _NoVisionModel:
        def encode_images(self, pixel_values, image_position_ids):
            raise AssertionError("must not be called when vision is not loaded")

    sched, _, _ = _fake_scheduler(_NoVisionModel())
    msg = UserMsg(
        uid=3,
        input_ids=torch.tensor([1, IMAGE_TOKEN_ID, 2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[b"unused-since-the-support-check-fires-first"],
    )

    error = Scheduler._fill_mm_embeds(sched, msg)

    assert error is not None
    assert "does not support image input" in error
    assert "3" in error  # names the request
    assert msg.mm_embeds is None


def test_fill_mm_embeds_errors_when_model_has_no_encode_images_at_all():
    sched, _, _ = _fake_scheduler(types.SimpleNamespace())  # no encode_images attribute
    msg = UserMsg(
        uid=4,
        input_ids=torch.tensor([1, IMAGE_TOKEN_ID, 2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[b"unused"],
    )

    error = Scheduler._fill_mm_embeds(sched, msg)

    assert error is not None
    assert "does not support image input" in error


def test_fill_mm_embeds_errors_on_slot_count_mismatch():
    """The embedding-rows-vs-placeholder-slots check this function exists to run BEFORE
    Qwen4ExpModel._merge_multimodal's own assertion -- with a message that says which
    request and what mismatched."""
    from freetoken.models.qwen4_exp.image import image_grid, num_image_tokens

    w, h = 320, 240
    n = num_image_tokens(image_grid(MODEL_PATH, w, h))

    class _BadVisionModel:
        vision_tower = object()

        def encode_images(self, pixel_values, image_position_ids):
            assert pixel_values.shape[0] == 1
            return torch.zeros(n - 1, 8)  # deliberately wrong row count

    sched, _, _ = _fake_scheduler(_BadVisionModel())
    msg = UserMsg(
        uid=5,
        input_ids=torch.tensor([1] + [IMAGE_TOKEN_ID] * n + [2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[_png_bytes(w, h, seed=1)],
    )

    error = Scheduler._fill_mm_embeds(sched, msg)

    assert error is not None
    assert "does not match" in error
    assert "5" in error
    assert msg.mm_embeds is None


def test_fill_mm_embeds_succeeds_and_sets_mm_embeds_when_counts_match():
    from freetoken.models.qwen4_exp.image import image_grid, num_image_tokens

    w, h = 320, 240
    n = num_image_tokens(image_grid(MODEL_PATH, w, h))

    class _VisionModel:
        vision_tower = object()

        def encode_images(self, pixel_values, image_position_ids):
            assert pixel_values.shape[0] == 1
            return torch.arange(n * 8, dtype=torch.float32).reshape(n, 8)

    sched, _, _ = _fake_scheduler(_VisionModel())
    msg = UserMsg(
        uid=6,
        input_ids=torch.tensor([1] + [IMAGE_TOKEN_ID] * n + [2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[_png_bytes(w, h, seed=2)],
    )

    error = Scheduler._fill_mm_embeds(sched, msg)

    assert error is None
    assert msg.mm_embeds is not None
    assert msg.mm_embeds.shape == (n, 8)


# --------------------------------------------------------------------------------------
# _process_one_msg wiring: the error path never reaches add_one_req; success does;
# text-only requests never touch _fill_mm_embeds at all (regression pin).
# --------------------------------------------------------------------------------------


def test_process_one_msg_rejects_and_does_not_admit_on_vision_error():
    class _NoVisionModel:
        def encode_images(self, *a, **k):
            raise AssertionError("must not be called")

    sched, added, sent = _fake_scheduler(_NoVisionModel())
    msg = UserMsg(
        uid=7,
        input_ids=torch.tensor([1, IMAGE_TOKEN_ID, 2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[b"irrelevant"],
    )

    Scheduler._process_one_msg(sched, msg)

    assert added == []
    assert len(sent) == 1
    assert isinstance(sent[0], ErrorReplyMsg)
    assert sent[0].uid == 7
    assert "does not support image input" in sent[0].error


def test_process_one_msg_admits_with_mm_embeds_on_success():
    from freetoken.models.qwen4_exp.image import image_grid, num_image_tokens

    w, h = 320, 240
    n = num_image_tokens(image_grid(MODEL_PATH, w, h))

    class _VisionModel:
        vision_tower = object()

        def encode_images(self, pixel_values, image_position_ids):
            return torch.zeros(n, 8)

    sched, added, sent = _fake_scheduler(_VisionModel())
    msg = UserMsg(
        uid=8,
        input_ids=torch.tensor([1] + [IMAGE_TOKEN_ID] * n + [2], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
        images=[_png_bytes(w, h, seed=3)],
    )

    Scheduler._process_one_msg(sched, msg)

    assert sent == []
    assert len(added) == 1 and added[0] is msg
    assert msg.mm_embeds is not None and msg.mm_embeds.shape == (n, 8)


def test_process_one_msg_text_only_never_calls_fill_mm_embeds():
    """Regression pin for 'the text-only path must be untouched in behaviour': a request
    with no images must admit exactly as before, without _fill_mm_embeds running at all
    (proven by making it raise if invoked, not just by it being a no-op)."""

    def _boom(self, msg):
        raise AssertionError("_fill_mm_embeds must not run for a text-only request")

    sched, added, sent = _fake_scheduler(model=SimpleNamespace())
    sched._fill_mm_embeds = types.MethodType(_boom, sched)
    msg = UserMsg(
        uid=9,
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
    )

    Scheduler._process_one_msg(sched, msg)

    assert sent == []
    assert added == [msg]
    assert msg.mm_embeds is None
