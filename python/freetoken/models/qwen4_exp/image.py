from __future__ import annotations

import functools
import io
import json
import os
from typing import Tuple, Union

import torch
from PIL import Image

# Image preprocessing adapter for Qwen4-Exp: turns PIL images / raw bytes / filesystem
# paths into the ``(pixel_values, image_position_ids)`` pair
# ``Qwen4ExpForCausalLM.encode_images`` (model.py) consumes.
#
# Preprocessing itself is NOT reimplemented here. The checkpoint's
# ``preprocessor_config.json`` declares ``image_processor_type: Qwen2VLImageProcessorFast``
# -- Qwen2-VL's resize/patchify, parameterised entirely by that config file. Two things
# about this environment matter for how the processor is constructed:
#   - ``transformers.AutoImageProcessor.from_pretrained(...)`` fails here: it demands
#     torchvision, which is not installed and must not be installed.
#   - Constructing ``Qwen2VLImageProcessor(**kwargs)`` directly from the config dict works
#     and transparently resolves to the torchvision-free ``Qwen2VLImageProcessorPil``
#     backend (``transformers.models.qwen2_vl.image_processing_pil_qwen2_vl``) instead --
#     same resize/patchify math, PIL-only implementation.
#
# The ``(row, col)`` convention ``image_position_ids`` must use is established from
# ``Qwen4VisionModel`` (vision.py), not assumed:
#   - ``_grid_extents``: ``rows = clamped[..., 0]``, ``cols = clamped[..., 1]``,
#     ``heights = rows.max()+1``, ``widths = cols.max()+1`` -- dim 0 is the height/row axis,
#     dim 1 is the width/col axis.
#   - ``_merge_group``: ``row, col = clamped[..., 0], clamped[..., 1]``;
#     ``block_index = (row // merge) * blocks_w + (col // merge)`` -- block-row-major over
#     merge blocks (row/height varies slower than col/width), matching a standard
#     height-major raster reading of "row" and "col".
# Separately, the processor's own flat ``pixel_values`` row order has a quirk: reading
# ``Qwen2VLImageProcessorPil.patchify`` (image_processing_pil_qwen2_vl.py), the resized
# image is reshaped to ``(C, GH, MH, PH, GW, MW, PW)`` (``GH``/``GW`` = merge-block grid,
# ``MH``/``MW`` = position within a merge block, ``PH``/``PW`` = pixel offset within a
# patch), transposed to ``(GH, GW, MH, MW, C, PH, PW)``, and flattened over the leading
# four axes -- so flat row ``i`` is NOT plain row-major raster order
# (``divmod(i, grid_w)``); it is block-row-major over 2x2 (``merge_size``) blocks, with the
# in-block row/col varying fastest. Verified empirically (a synthetic image tagging every
# patch with its own ``(row, col)`` recovers this block order, not naive row-major, when
# run through the real ``patchify()``) and algebraically (this is exactly the ordering
# ``_merge_group`` expects: its ``target = block_index * merge**2 + sub_index`` equals the
# processor's flat index ``i`` term for term).
#
# Rather than re-deriving that block-major arithmetic a second time, this module builds
# ``image_position_ids`` with ``transformers.vision_utils.get_vision_position_ids`` -- HF's
# own helper for exactly this coordinate assignment (also used as the ground-truth position
# generator in ``tests/models/test_qwen4_exp_vision.py``'s HF-parity tests). Its
# construction (``hpos_ids/wpos_ids = meshgrid(...).reshape(h//m, m, w//m,
# m).transpose(1, 2).flatten()``, stacked as ``(hpos, wpos)``) is the same block-major
# order derived above, confirming both readings agree.


ImageLike = Union[Image.Image, bytes, bytearray, str, "os.PathLike[str]"]

_PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"
_DROP_KEYS = ("processor_class", "image_processor_type", "auto_map")


@functools.cache
def load_image_processor(model_path: str):
    """Builds a Qwen2-VL image processor from ``<model_path>/preprocessor_config.json``.

    Cached per ``model_path`` (same idiom as ``freetoken.utils.hf._load_hf_config``) since
    every caller in this module needs one and construction re-parses the config file
    otherwise. Raises ``FileNotFoundError`` with a clear message if the checkpoint carries
    no ``preprocessor_config.json``.
    """
    config_path = os.path.join(model_path, _PREPROCESSOR_CONFIG_NAME)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"no {_PREPROCESSOR_CONFIG_NAME} under {model_path!r} -- Qwen4-Exp image "
            "preprocessing needs this file to build the Qwen2-VL processor"
        )
    from transformers import Qwen2VLImageProcessor

    with open(config_path) as f:
        raw = json.load(f)
    kwargs = {k: v for k, v in raw.items() if k not in _DROP_KEYS}
    return Qwen2VLImageProcessor(**kwargs)


def image_grid(model_path: str, width: int, height: int) -> Tuple[int, int, int]:
    """Patch grid ``(t, h, w)`` for an image of the given pixel size, without decoding
    pixels. ``t`` is always 1 for a still image -- the temporal axis exists in the shared
    Qwen2-VL processor for video input; this adapter is images-only.

    Uses ``transformers``' own ``smart_resize`` (the exact resize-target arithmetic the
    processor's ``resize()`` calls internally), fed the loaded processor's own
    ``patch_size``/``merge_size``/``size`` -- not a re-derivation of that formula -- so this
    always agrees with what the full pixel pipeline (``preprocess_images``) would produce
    for the same image (pinned by
    ``tests/models/test_qwen4_exp_image.py::test_image_grid_matches_processor``).
    """
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

    proc = load_image_processor(model_path)
    factor = proc.patch_size * proc.merge_size
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=proc.size["shortest_edge"],
        max_pixels=proc.size["longest_edge"],
    )
    return 1, resized_h // proc.patch_size, resized_w // proc.patch_size


def num_image_tokens(grid: Tuple[int, int, int], merge_size: int = 2) -> int:
    """How many ``<|image_pad|>`` placeholder tokens one image occupies: the
    merged-resolution token count, i.e. the grid's patch count divided by
    ``merge_size**2``.

    Contract: this is the number of placeholder tokens the TEXT side (tokenizer /
    prompt-builder) must emit for an image with this patch grid -- if it emits a different
    count, ``Qwen4ExpModel._merge_multimodal`` (model.py) asserts at merge time, since the
    placeholder-slot count and ``mm_embeds.shape[0]`` (which is exactly this many rows,
    times number of images) must match exactly.

    ``merge_size`` defaults to 2, this checkpoint's ``spatial_merge_size`` (also the
    Qwen2-VL processor's own default ``merge_size``); pass the loaded processor's
    ``.merge_size`` explicitly if it might ever differ.
    """
    t, h, w = grid
    assert h % merge_size == 0 and w % merge_size == 0, (
        f"grid {grid} is not divisible by merge_size {merge_size}"
    )
    return t * (h // merge_size) * (w // merge_size)


def _load_one(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(os.fspath(image)).convert("RGB")


def preprocess_images(
    model_path: str, images: list[ImageLike]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs the Qwen2-VL processor and adapts its flat layout to ``encode_images``'s
    per-image, padded contract.

    Accepts PIL images, raw encoded image bytes, or filesystem paths (any mix, in one
    call). Returns:
      - ``pixel_values``: ``[num_images, max_patches, in_channels*temporal_patch_size*patch_size**2]``
      - ``image_position_ids``: ``[num_images, max_patches, 2]``, 0-indexed pre-merge
        ``(row, col)`` patch-grid coordinates, ``(-1, -1)`` padding for images shorter than
        the batch's longest -- the exact contract ``Qwen4VisionModel.forward`` /
        ``Qwen4ExpForCausalLM.encode_images`` documents.

    The processor itself returns a flat ``pixel_values [total_patches, patch_dim]`` plus
    ``image_grid_thw [n_images, 3]``. This function splits the flat rows back out per image
    (using each image's ``t*h*w`` patch count from ``image_grid_thw``), pads every image to
    the batch's longest patch count, and derives the matching ``(row, col)`` coordinates
    with ``transformers.vision_utils.get_vision_position_ids`` -- see the module docstring
    for why that (not a hand-rolled ``divmod``) is the right ordering.
    """
    from transformers.vision_utils import get_vision_position_ids

    proc = load_image_processor(model_path)
    pil_images = [_load_one(im) for im in images]
    out = proc(images=pil_images, return_tensors="pt")
    flat_pixel_values: torch.Tensor = out["pixel_values"]  # [total_patches, patch_dim]
    grid_thw: torch.Tensor = out["image_grid_thw"]  # [n_images, 3]

    counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    n_images = len(counts)
    max_patches = max(counts) if counts else 0
    patch_dim = flat_pixel_values.shape[-1]

    pixel_values = flat_pixel_values.new_zeros(n_images, max_patches, patch_dim)
    image_position_ids = flat_pixel_values.new_full(
        (n_images, max_patches, 2), -1, dtype=torch.long
    )

    offset = 0
    for i, n in enumerate(counts):
        pixel_values[i, :n] = flat_pixel_values[offset : offset + n]
        image_position_ids[i, :n] = get_vision_position_ids(grid_thw[i : i + 1], proc.merge_size)
        offset += n

    return pixel_values, image_position_ids


__all__ = [
    "load_image_processor",
    "image_grid",
    "num_image_tokens",
    "preprocess_images",
]
