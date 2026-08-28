"""Tests for Qwen4-Exp's image preprocessing adapter (`freetoken.models.qwen4_exp.image`)
and the `_merge_multimodal` splice `freetoken.models.qwen4_exp.model.Qwen4ExpModel` wires
image embeddings through.

Most tests here build a synthetic `preprocessor_config.json` (`PREPROCESSOR_CONFIG` below,
copied verbatim from the real checkpoint's, see the module docstring in `image.py`) in a
`tmp_path` so they run without the real checkpoint on disk -- `load_image_processor` only
ever reads that one file. A `slow`, `skipif`-gated test at the bottom re-runs the same
checks against the REAL `preprocessor_config.json` on disk, to catch drift between the
fabricated fixture and the actual checkpoint (matching the skip style
`tests/models/test_qwen4_exp_vision.py` uses for its real-checkpoint tests).

No GPU, no server, no model weights: everything here is CPU-only image/tensor plumbing.
"""

from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest
import torch
from PIL import Image

MODEL_PATH = "/root/models/Qwen3.8-Flash-Next-FP8"

# Copied verbatim from MODEL_PATH/preprocessor_config.json (see qwen4_exp/image.py's module
# docstring for how this environment constructs a processor from it without torchvision).
PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


@pytest.fixture()
def fake_model_dir(tmp_path) -> str:
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG))
    return str(tmp_path)


def _synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# (width, height) pairs: square + non-square + aspect-ratio variety, including sizes whose
# dimensions are NOT multiples of patch_size*merge_size (16*2 == 32) so smart_resize's
# rounding is actually exercised, not sidestepped.
SIZES = [
    (320, 240),  # 240 not a multiple of 32
    (500, 333),  # neither dimension a multiple of 32, non-square
    (288, 288),  # both multiples of 32, square control case
]


# --------------------------------------------------------------------------------------
# image_grid / num_image_tokens: the tokenizer/core contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", SIZES)
def test_image_grid_matches_full_processor(fake_model_dir, width, height):
    """`image_grid` (header-only, no pixel decode) must agree exactly with the
    `image_grid_thw` the full pixel pipeline produces for the same image -- if these ever
    disagree, the model asserts at merge time (see `image.py`'s `num_image_tokens`
    docstring), so this is pinned directly rather than assumed."""
    from freetoken.models.qwen4_exp.image import image_grid, load_image_processor

    grid = image_grid(fake_model_dir, width, height)

    proc = load_image_processor(fake_model_dir)
    img = _synthetic_image(width, height, seed=width * 1000 + height)
    out = proc(images=img, return_tensors="pt")
    expected = tuple(out["image_grid_thw"][0].tolist())

    assert grid == expected, f"image_grid{(width, height)}={grid} != processor's {expected}"


@pytest.mark.parametrize("width,height", SIZES)
def test_num_image_tokens_equals_merge_block_count(fake_model_dir, width, height):
    """`num_image_tokens` must equal the number of 2x2 merge blocks -- cross-checked two
    ways: (1) directly against `(h // merge) * (w // merge)` and (2) against the number of
    distinct merge blocks the ACTUAL `(row, col)` position ids from `preprocess_images`
    fall into, so the token count and the position-id geometry `_merge_multimodal`'s
    caller relies on cannot silently disagree."""
    from freetoken.models.qwen4_exp.image import (
        image_grid,
        num_image_tokens,
        preprocess_images,
    )

    grid = image_grid(fake_model_dir, width, height)
    t, h, w = grid
    merge = PREPROCESSOR_CONFIG["merge_size"]
    expected = t * (h // merge) * (w // merge)

    n = num_image_tokens(grid, merge_size=merge)
    assert n == expected

    img = _synthetic_image(width, height, seed=width + height)
    _, position_ids = preprocess_images(fake_model_dir, [img])
    valid = position_ids[0]
    valid = valid[(valid != -1).all(dim=-1)]
    blocks = {(int(r) // merge, int(c) // merge) for r, c in valid.tolist()}
    assert len(blocks) == n, f"distinct merge blocks ({len(blocks)}) != num_image_tokens ({n})"


# --------------------------------------------------------------------------------------
# preprocess_images: round-trip and multi-image batching
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", SIZES)
def test_round_trip_single_image(fake_model_dir, width, height):
    """Single-image round trip: output shapes match `encode_images`'s contract
    (`[num_images, num_patches, patch_dim]` / `[num_images, num_patches, 2]`), the rows are
    byte-identical (not approximately) to an independent processor call on the same image,
    and the `(row, col)` coordinates cover exactly `[0, h) x [0, w)` with no repeats."""
    from freetoken.models.qwen4_exp.image import load_image_processor, preprocess_images

    img = _synthetic_image(width, height, seed=100 + width - height)
    proc = load_image_processor(fake_model_dir)
    ref = proc(images=img, return_tensors="pt")
    ref_pixel_values = ref["pixel_values"]
    t, h, w = ref["image_grid_thw"][0].tolist()
    patch_dim = ref_pixel_values.shape[-1]
    assert patch_dim == 3 * PREPROCESSOR_CONFIG["temporal_patch_size"] * PREPROCESSOR_CONFIG["patch_size"] ** 2

    pixel_values, position_ids = preprocess_images(fake_model_dir, [img])

    assert pixel_values.shape == (1, h * w, patch_dim)
    assert position_ids.shape == (1, h * w, 2)
    # No padding at all for a single image -- every row is "real".
    assert torch.equal(pixel_values[0], ref_pixel_values), "pixel rows are not byte-identical to the processor"
    assert bool((position_ids[0] != -1).all()), "single-image call must have no (-1, -1) padding"

    rows_cols = {(int(r), int(c)) for r, c in position_ids[0].tolist()}
    assert rows_cols == {(r, c) for r in range(h) for c in range(w)}, (
        "position ids must cover every (row, col) in the patch grid exactly once"
    )


def test_multi_image_batch_padding(fake_model_dir):
    """Two differently-sized images in one `preprocess_images` call pad to a common patch
    count; each image's real rows survive intact (byte-identical to what preprocessing that
    image alone would produce) and the padding tail carries `(-1, -1)`."""
    from freetoken.models.qwen4_exp.image import load_image_processor, preprocess_images

    (w0, h0), (w1, h1) = SIZES[0], SIZES[1]
    img0 = _synthetic_image(w0, h0, seed=11)
    img1 = _synthetic_image(w1, h1, seed=22)

    proc = load_image_processor(fake_model_dir)
    ref0 = proc(images=img0, return_tensors="pt")
    ref1 = proc(images=img1, return_tensors="pt")
    n0 = int(ref0["image_grid_thw"][0].prod())
    n1 = int(ref1["image_grid_thw"][0].prod())
    assert n0 != n1, "test fixture sizes must produce different patch counts to exercise padding"

    pixel_values, position_ids = preprocess_images(fake_model_dir, [img0, img1])

    max_patches = max(n0, n1)
    assert pixel_values.shape == (2, max_patches, ref0["pixel_values"].shape[-1])
    assert position_ids.shape == (2, max_patches, 2)

    for i, (ref, n) in enumerate([(ref0, n0), (ref1, n1)]):
        assert torch.equal(pixel_values[i, :n], ref["pixel_values"]), f"image {i}: real rows corrupted by batching"
        if n < max_patches:
            pad_pixels = pixel_values[i, n:]
            pad_pos = position_ids[i, n:]
            assert bool((pad_pos == -1).all()), f"image {i}: padding position ids must be (-1, -1)"
            assert bool((pad_pixels == 0).all()), f"image {i}: padding pixel rows must be zero-filled"


def test_preprocess_images_accepts_bytes_and_path(fake_model_dir, tmp_path):
    """Accepts PIL images, raw encoded bytes, and filesystem paths -- and all three must
    agree on the same underlying image."""
    from freetoken.models.qwen4_exp.image import preprocess_images

    w, h = SIZES[0]
    img = _synthetic_image(w, h, seed=7)

    path = tmp_path / "img.png"
    img.save(path)
    buf = tmp_path / "img_bytes.png"
    img.save(buf)
    raw_bytes = buf.read_bytes()

    pv_pil, pid_pil = preprocess_images(fake_model_dir, [img])
    pv_bytes, pid_bytes = preprocess_images(fake_model_dir, [raw_bytes])
    pv_path, pid_path = preprocess_images(fake_model_dir, [str(path)])

    # PNG is lossless, so round-tripping through disk/bytes must reproduce identical pixels.
    assert torch.equal(pv_pil, pv_bytes)
    assert torch.equal(pv_pil, pv_path)
    assert torch.equal(pid_pil, pid_bytes)
    assert torch.equal(pid_pil, pid_path)


# --------------------------------------------------------------------------------------
# The splice: _merge_multimodal
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _global_ctx_guard():
    """Mirrors `tests/models/test_qwen4_ple_ngram_hoist.py`'s guard: save/restore the
    process-global context so this module's tests can't leak state into others."""
    import freetoken.core as core

    saved = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = saved


def _make_ctx_with_batch(mm_embeds):
    import freetoken.core as core
    from freetoken.core import Batch, Context

    ctx = Context(page_size=1)
    core._GLOBAL_CTX = ctx
    batch = Batch(reqs=[], phase="prefill")
    batch.mm_embeds = mm_embeds
    return ctx, batch


def test_merge_multimodal_writes_exactly_the_image_token_rows():
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    hidden = 8
    seq_len = 10
    image_token_id = 999
    image_positions = [2, 3, 6]

    input_ids = torch.tensor(
        [image_token_id if i in image_positions else 100 + i for i in range(seq_len)],
        dtype=torch.long,
    )
    x = torch.randn(seq_len, hidden)
    x_before = x.clone()
    mm_embeds = torch.arange(len(image_positions) * hidden, dtype=torch.float32).reshape(
        len(image_positions), hidden
    )

    fake_self = types.SimpleNamespace(_image_token_id=image_token_id)
    ctx, batch = _make_ctx_with_batch(mm_embeds)

    with ctx.forward_batch(batch):
        out = Qwen4ExpModel._merge_multimodal(fake_self, input_ids, x)

    for slot, pos in enumerate(image_positions):
        assert torch.equal(out[pos], mm_embeds[slot]), f"row {pos} was not overwritten with mm_embeds[{slot}]"
    for pos in range(seq_len):
        if pos not in image_positions:
            assert torch.equal(out[pos], x_before[pos]), f"row {pos} was modified but is not an image-token slot"


def test_merge_multimodal_asserts_on_slot_count_mismatch():
    """A malformed batch (wrong number of image-token slots vs. vision features) must fail
    loudly, not silently misplace embeddings -- this is the invariant
    `scheduler/scheduler.py`'s `_gather_multimodal` depends on."""
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    hidden = 4
    input_ids = torch.tensor([1, 999, 2, 999], dtype=torch.long)  # 2 image-token slots
    x = torch.randn(4, hidden)
    mm_embeds = torch.randn(3, hidden)  # 3 != 2

    fake_self = types.SimpleNamespace(_image_token_id=999)
    ctx, batch = _make_ctx_with_batch(mm_embeds)

    with ctx.forward_batch(batch), pytest.raises(AssertionError):
        Qwen4ExpModel._merge_multimodal(fake_self, input_ids, x)


def test_merge_multimodal_no_op_when_batch_has_no_images():
    """Text-only batches (`mm_embeds is None`, the production default) must be an EXACT
    no-op -- same object back, not just an equal-valued copy."""
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    x = torch.randn(4, 6)

    fake_self = types.SimpleNamespace(_image_token_id=999)
    ctx, batch = _make_ctx_with_batch(mm_embeds=None)

    with ctx.forward_batch(batch):
        out = Qwen4ExpModel._merge_multimodal(fake_self, input_ids, x)

    assert out is x, "no-mm_embeds path must return the same tensor object, not a copy"


def test_merge_multimodal_no_op_when_model_is_not_multimodal():
    """A text-only checkpoint (`image_token_id is None`, e.g. `is_multimodal=False` /
    vision not loaded) must also be an exact no-op even if a batch somehow carried
    `mm_embeds` -- production `ft serve` boot never sets `image_token_id` unless vision is
    opted in (see `qwen4_exp/config.py`'s `_parse_vision_config`)."""
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    x = torch.randn(4, 6)

    fake_self = types.SimpleNamespace(_image_token_id=None)
    ctx, batch = _make_ctx_with_batch(mm_embeds=torch.randn(2, 6))

    with ctx.forward_batch(batch):
        out = Qwen4ExpModel._merge_multimodal(fake_self, input_ids, x)

    assert out is x


# --------------------------------------------------------------------------------------
# Real-checkpoint cross-check: proves PREPROCESSOR_CONFIG above hasn't drifted from the
# actual shipped file, and that the whole adapter works unmodified against it.
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH), reason="requires the real Qwen3.8-Flash-Next-FP8 checkpoint on disk"
)
@pytest.mark.slow
def test_real_checkpoint_preprocessor_config_matches_fixture():
    from freetoken.models.qwen4_exp.image import (
        image_grid,
        load_image_processor,
        num_image_tokens,
        preprocess_images,
    )

    with open(os.path.join(MODEL_PATH, "preprocessor_config.json")) as f:
        real = json.load(f)
    assert real == PREPROCESSOR_CONFIG, (
        "PREPROCESSOR_CONFIG in this test file has drifted from the real checkpoint's "
        f"preprocessor_config.json: real={real}"
    )

    proc = load_image_processor(MODEL_PATH)
    assert (proc.patch_size, proc.merge_size, proc.temporal_patch_size) == (16, 2, 2)

    # The exact case documented in the assignment: a 640x480 RGB image.
    img = _synthetic_image(640, 480, seed=640480)
    grid = image_grid(MODEL_PATH, 640, 480)
    assert grid == (1, 30, 40)
    assert num_image_tokens(grid, merge_size=proc.merge_size) == 300

    pixel_values, position_ids = preprocess_images(MODEL_PATH, [img])
    assert pixel_values.shape == (1, 1200, 1536)
    assert position_ids.shape == (1, 1200, 2)
