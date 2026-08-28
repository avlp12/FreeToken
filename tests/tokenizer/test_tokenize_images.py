"""Pad-expansion coverage: `TokenizeManager.tokenize` must turn the chat template's
single `<|image_pad|>` per image into `num_image_tokens(image_grid(...))` copies, sized
from each image's header (no pixel decode), in the order the images appear -- the
text-side half of the placeholder-count contract `Qwen4ExpModel._merge_multimodal`
(model.py) asserts at merge time.

Uses the real Qwen3.8-Flash-Next-FP8 checkpoint on disk (config.json's image_token_id,
preprocessor_config.json's resize/patch math) so the expected counts are not
re-derived by hand; skipped when that checkpoint is not present. CPU-only: PIL header
reads and freetoken.models.qwen4_exp.image's processor construction, no torch model,
no GPU.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager

MODEL_PATH = "/root/models/Qwen3.8-Flash-Next-FP8"
VISION_START = 248053
IMAGE_PAD = 248056
VISION_END = 248054

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH), reason="requires the real Qwen3.8-Flash-Next-FP8 checkpoint on disk"
)


def _png_bytes(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class _FakeTokenizer:
    """Fakes only what TokenizeManager needs: name_or_path (to resolve the checkpoint
    for image_grid / image_token_id) and a canned rendered token-id sequence standing
    in for what apply_chat_template + encode would really produce -- the chat template
    itself (chat_template.jinja) is exercised separately; this isolates the pad-count
    arithmetic."""

    def __init__(self, rendered_ids: list[int]) -> None:
        self.name_or_path = MODEL_PATH
        self.chat_template = "irrelevant-non-empty-marker"
        self._rendered_ids = rendered_ids

    def apply_chat_template(self, messages, **kwargs):
        return "rendered-prompt"

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        assert prompt == "rendered-prompt"
        return torch.tensor([self._rendered_ids], dtype=torch.long)


def _msg(images: list[bytes] | None) -> TokenizeMsg:
    # The exact content doesn't matter -- _FakeTokenizer.apply_chat_template ignores it
    # and returns a canned "rendered-prompt" string; what's under test is what
    # TokenizeManager does with the (also canned) encoded token ids and msg.images.
    return TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        sampling_params=SamplingParams(),
        images=images,
    )


def test_two_images_in_one_message_expand_to_the_right_counts_and_positions():
    from freetoken.models.qwen4_exp.image import image_grid, num_image_tokens

    (w0, h0), (w1, h1) = (320, 240), (500, 333)
    n0 = num_image_tokens(image_grid(MODEL_PATH, w0, h0))
    n1 = num_image_tokens(image_grid(MODEL_PATH, w1, h1))
    assert n0 != n1, "fixture sizes must differ in patch count to prove per-image sizing"

    rendered = (
        [10, 11]
        + [VISION_START, IMAGE_PAD, VISION_END]
        + [12]
        + [VISION_START, IMAGE_PAD, VISION_END]
        + [13, 14]
    )
    tokenizer = _FakeTokenizer(rendered)
    manager = TokenizeManager(tokenizer)
    images = [_png_bytes(w0, h0, seed=1), _png_bytes(w1, h1, seed=2)]

    [input_ids] = manager.tokenize([_msg(images)])
    ids = input_ids.tolist()

    expected = (
        [10, 11]
        + [VISION_START] + [IMAGE_PAD] * n0 + [VISION_END]
        + [12]
        + [VISION_START] + [IMAGE_PAD] * n1 + [VISION_END]
        + [13, 14]
    )
    assert ids == expected
    # Every expanded run is still bracketed by vision_start/vision_end, and only by
    # image_pad in between -- no stray tokens leaked into the run.
    run0 = ids[2 : 2 + 1 + n0 + 1]
    assert run0[0] == VISION_START and run0[-1] == VISION_END
    assert run0[1:-1] == [IMAGE_PAD] * n0


def test_single_image_expansion_matches_num_image_tokens():
    from freetoken.models.qwen4_exp.image import image_grid, num_image_tokens

    w, h = 288, 288
    n = num_image_tokens(image_grid(MODEL_PATH, w, h))
    rendered = [1] + [VISION_START, IMAGE_PAD, VISION_END] + [2]
    tokenizer = _FakeTokenizer(rendered)
    manager = TokenizeManager(tokenizer)

    [input_ids] = manager.tokenize([_msg([_png_bytes(w, h, seed=3)])])

    assert input_ids.tolist() == [1, VISION_START] + [IMAGE_PAD] * n + [VISION_END, 2]


def test_pad_image_count_mismatch_raises_clean_error():
    """One image but two placeholders (or vice versa) is a server-side rendering bug or
    a malformed request either way -- must fail loudly with a clear message, not
    silently misalign images to placeholders."""
    rendered = [1, VISION_START, IMAGE_PAD, VISION_END, 2]
    tokenizer = _FakeTokenizer(rendered)
    manager = TokenizeManager(tokenizer)
    images = [b"image-one", b"image-two"]  # 2 images, but only 1 placeholder

    with pytest.raises(ValueError, match="image"):
        manager.tokenize([_msg(images)])


def test_no_expansion_when_message_carries_no_images():
    """Regression pin: a message with images=None (the text-only default) must not
    touch the token stream at all, even if it happens to contain the pad token id."""
    rendered = [1, IMAGE_PAD, 2]
    tokenizer = _FakeTokenizer(rendered)
    manager = TokenizeManager(tokenizer)

    [input_ids] = manager.tokenize([_msg(images=None)])

    assert input_ids.tolist() == [1, IMAGE_PAD, 2]
