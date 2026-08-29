"""Structural coverage for the vision tower (`freetoken.models.qwen4_exp.vision`) and its
weight-loading seam (`adapt_vision_tensor` / `weight.py`'s ``_rename``).

The forward-pass tests use a tiny synthetic ``VisionConfig`` (small dims, CPU, no GPU) to
pin the module's shape contract cheaply. The bottom, ``skipif``-gated test cross-checks
the module built from the REAL checkpoint's ``vision_config`` against every tensor name
and shape ``model.safetensors.index.json`` actually ships under ``model.visual.*`` --
catching drift between this port and the checkpoint without needing a GPU or a full
model load (only the one safetensors shard the vision tensors live in is read).
"""

from __future__ import annotations

import os

import pytest
import safetensors
import torch

from freetoken.models.qwen4_exp.config import VisionConfig
from freetoken.models.qwen4_exp.vision import Qwen4VisionModel, adapt_vision_tensor

MODEL_PATH = "/root/models/Qwen3.8-Flash-Next-NVFP4"


def _tiny_vision_config(**overrides) -> VisionConfig:
    base = dict(
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,  # 4x4 grid
        out_hidden_size=8,
        hidden_act="gelu_pytorch_tanh",
        rope_theta=10000.0,
        text_hidden_size=8,
        deepstack_visual_indexes=(),
    )
    base.update(overrides)
    return VisionConfig(**base)


def _randomize_weights(model: Qwen4VisionModel) -> None:
    """BaseOP submodules allocate their parameters with ``torch.empty`` (this runtime
    only ever fills them from a checkpoint's ``load_state_dict``, never trains) -- a
    bare-constructed module's forward pass would run over uninitialized memory
    (frequently NaN/Inf by chance). Fill every tensor with real values so a shape-only
    forward test actually exercises finite math."""
    for tensor in model.state_dict().values():
        tensor.normal_()


def test_forward_single_image_no_padding():
    vc = _tiny_vision_config()
    model = Qwen4VisionModel(vc)
    _randomize_weights(model)
    merge = vc.spatial_merge_size
    h = w = 4  # one 2x2-merge-block grid, no padding
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2

    pixel_values = torch.randn(1, h * w, patch_dim)
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    position_ids = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=-1).unsqueeze(0)

    out = model.forward(pixel_values, position_ids)

    expected_tokens = (h // merge) * (w // merge)
    assert out.shape == (expected_tokens, vc.out_hidden_size)
    assert torch.isfinite(out).all()


def test_forward_batches_two_images_with_padding():
    """A smaller second image pads to the batch's longest patch count; the padded rows
    must be excluded from the returned soft tokens (only real merge blocks come out)."""
    vc = _tiny_vision_config()
    merge = vc.spatial_merge_size
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    model = Qwen4VisionModel(vc)
    _randomize_weights(model)

    h0, w0 = 4, 4  # 4 merge blocks
    h1, w1 = 2, 2  # 1 merge block
    n0, n1 = h0 * w0, h1 * w1
    max_patches = max(n0, n1)

    pixel_values = torch.zeros(2, max_patches, patch_dim)
    position_ids = torch.full((2, max_patches, 2), -1, dtype=torch.long)
    for i, (h, w, n) in enumerate([(h0, w0, n0), (h1, w1, n1)]):
        pixel_values[i, :n] = torch.randn(n, patch_dim)
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        position_ids[i, :n] = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=-1)

    out = model.forward(pixel_values, position_ids)

    expected_tokens = (h0 // merge) * (w0 // merge) + (h1 // merge) * (w1 // merge)
    assert out.shape == (expected_tokens, vc.out_hidden_size)


def test_deepstack_visual_indexes_not_implemented():
    vc = _tiny_vision_config(deepstack_visual_indexes=(3,))
    with pytest.raises(NotImplementedError, match="deepstack"):
        Qwen4VisionModel(vc)


def test_adapt_vision_tensor_reshapes_only_the_patch_embed_conv3d_weight():
    conv3d_weight = torch.randn(16, 3, 2, 4, 4)  # [C_out, C_in, T, P, P]
    reshaped = adapt_vision_tensor("patch_embed.proj.weight", conv3d_weight)
    assert reshaped.shape == (16, 3 * 2 * 4 * 4)
    assert torch.equal(reshaped, conv3d_weight.reshape(16, -1))

    other = torch.randn(16, 16)
    assert adapt_vision_tensor("blocks.0.norm1.weight", other) is other


# --------------------------------------------------------------------------------------
# Real-checkpoint cross-check: every model.visual.* tensor the checkpoint ships has a
# same-named, same-shaped home in Qwen4VisionModel's own state dict, and vice versa.
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH), reason="requires the real Qwen3.8-Flash-Next-NVFP4 checkpoint on disk"
)
@pytest.mark.slow
def test_real_checkpoint_vision_tensors_match_module_state_dict(monkeypatch):
    import json

    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.weight import _rename
    from freetoken.utils import cached_load_hf_config

    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    cfg = parse_config(cached_load_hf_config(MODEL_PATH))
    assert cfg.is_multimodal, "the NVFP4 checkpoint ships a vision_config"
    expected = Qwen4VisionModel(cfg.vision_config).state_dict()

    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    vision_files = sorted({fn for k, fn in weight_map.items() if k.startswith("model.visual.")})

    produced: dict[str, torch.Tensor] = {}
    for fname in vision_files:
        with safetensors.safe_open(
            os.path.join(MODEL_PATH, fname), framework="pt", device="cpu"
        ) as f:
            for raw_name in f.keys():
                if not raw_name.startswith(("model.visual.", "visual.")):
                    continue
                name = _rename(raw_name)
                assert name is not None and name.startswith("vision_tower.")
                short = name[len("vision_tower."):]
                produced[short] = adapt_vision_tensor(short, f.get_tensor(raw_name))

    assert set(produced) == set(expected), (
        f"missing={sorted(set(expected) - set(produced))[:10]} "
        f"extra={sorted(set(produced) - set(expected))[:10]}"
    )
    mismatched = [
        k for k in expected if tuple(expected[k].shape) != tuple(produced[k].shape)
    ]
    assert not mismatched, f"shape mismatches: {mismatched[:10]}"
