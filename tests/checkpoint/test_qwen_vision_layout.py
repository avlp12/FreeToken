"""Coverage for the Qwen4-Exp FTW converter now carrying the vision tower instead of
dropping it (``checkpoint/qwen_layout.py`` + ``checkpoint/convert.py``'s dedicated
``_iter_vision_from_disk`` pass-through).

Two tiers of evidence, both required (see the assignment this closes):

* Cheap, index-only tests (no torch, no safetensors data read): classify every one of the
  real checkpoint's 333 ``model.visual.*`` keys via ``qwen_layout.classify_tensor`` /
  ``is_vision_tensor`` / ``ftw_vision_name`` and prove the renamed set is in EXACT
  bijection with ``Qwen4VisionModel(vision_config).state_dict()``'s keys (prefixed
  ``vision_tower.``) -- a missing or extra component always leaves leftover tensors on one
  side or the other, so set equality is the strongest structural evidence available
  offline, without reconverting anything.
* One slower, real-data test (``pytest.mark.slow``, same convention as
  ``tests/models/test_qwen4_exp_vision.py``) that exercises convert.py's ACTUAL
  ``_iter_vision_from_disk`` generator end to end (no FTWWriter, no disk write -- just the
  generator) and checks the renamed ``patch_embed.proj.weight`` lands at the reshaped
  Linear-equivalent shape ``[1152, 1536]``.

All real-checkpoint tests are ``skipif``-guarded on the source checkpoint being present on
disk, matching ``tests/models/test_qwen4_exp_vision.py``'s style.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

MODEL_PATH = "/root/models/Qwen3.8-Flash-Next-FP8"
_HAVE_MODEL = os.path.isdir(MODEL_PATH)

pytestmark = pytest.mark.skipif(
    not _HAVE_MODEL, reason="requires the real Qwen3.8-Flash-Next-FP8 checkpoint on disk"
)


# ---------------------------------------------------------------------------
# Cheap, index-only tests (real index.json, no safetensors data read).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_vision_keys() -> set[str]:
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    keys = {k for k in weight_map if k.startswith("model.visual.")}
    assert len(keys) == 333, f"expected 333 model.visual.* keys, found {len(keys)}"
    return keys


@pytest.fixture(scope="module")
def real_vision_config():
    """The checkpoint's real ``VisionConfig``, via the actual ``parse_config`` code path
    (not hand-built) -- forces ``FREETOKEN_LOAD_VISION=1`` for the duration of this
    fixture only, regardless of the ambient env, since this is a structural check, not a
    test of the opt-in gate itself (that is covered by
    ``tests/models/test_qwen4_exp_vision.py::test_parse_config_wires_vision_config_only_when_opted_in``).
    """
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.utils.hf import cached_load_hf_config

    old = os.environ.get("FREETOKEN_LOAD_VISION")
    os.environ["FREETOKEN_LOAD_VISION"] = "1"
    try:
        cfg = parse_config(cached_load_hf_config(MODEL_PATH))
    finally:
        if old is None:
            os.environ.pop("FREETOKEN_LOAD_VISION", None)
        else:
            os.environ["FREETOKEN_LOAD_VISION"] = old
    assert cfg.vision_config is not None
    return cfg.vision_config


def test_real_vision_keys_classify_as_carried_not_skipped(real_vision_keys):
    """Every one of the 333 real ``model.visual.*`` keys must classify as carried
    (DEST_DENSE / kind="weight") -- NOT DEST_SKIP. Before this change all 333 were
    DEST_SKIP (see the production FTW's 0 visual tensors)."""
    from freetoken.checkpoint.qwen_layout import DEST_DENSE, DEST_SKIP, classify_tensor, is_vision_tensor

    non_skip = 0
    for name in real_vision_keys:
        assert is_vision_tensor(name), name
        dest = classify_tensor(name)
        assert dest != DEST_SKIP, f"{name} still classifies as DEST_SKIP"
        assert dest == DEST_DENSE, f"{name} classified as {dest!r}, expected DEST_DENSE"
        non_skip += 1
    assert non_skip == 333


def test_real_vision_renamed_set_equals_model_state_dict_exact(real_vision_keys, real_vision_config):
    """Exact set equality: {ftw_vision_name(k) for k in the 333 real checkpoint keys} ==
    {"vision_tower." + k for k in Qwen4VisionModel(vision_config).state_dict()}. Zero
    missing, zero unexpected on either side -- a missing/extra module component always
    shows up as a leftover here."""
    from freetoken.checkpoint.qwen_layout import ftw_vision_name
    from freetoken.models.qwen4_exp.vision import Qwen4VisionModel

    renamed = {ftw_vision_name(k) for k in real_vision_keys}
    assert len(renamed) == 333, "rename collided two distinct checkpoint keys onto one name"

    model = Qwen4VisionModel(real_vision_config)
    expected = {"vision_tower." + k for k in model.state_dict().keys()}

    missing = expected - renamed
    unexpected = renamed - expected
    assert not missing and not unexpected, (
        f"renamed vision keys != Qwen4VisionModel.state_dict() keys: "
        f"missing(in model, not in checkpoint)={missing}, "
        f"unexpected(in checkpoint, not in model)={unexpected}"
    )
    assert renamed == expected


def test_real_vision_renamed_names_match_vision_key_prefixes(real_vision_keys):
    """Every renamed name must start with a ``models.config.VISION_KEY_PREFIXES`` entry --
    that is exactly what ``models/weight.py``'s FTW load-time filter matches against to
    skip vision when ``FREETOKEN_LOAD_VISION`` is off."""
    from freetoken.checkpoint.qwen_layout import ftw_vision_name
    from freetoken.models.config import VISION_KEY_PREFIXES

    for name in real_vision_keys:
        renamed = ftw_vision_name(name)
        assert renamed.startswith(VISION_KEY_PREFIXES), (
            f"{renamed!r} (from {name!r}) does not start with any of {VISION_KEY_PREFIXES}"
        )


def test_ftw_vision_name_raises_on_non_vision_key():
    from freetoken.checkpoint.qwen_layout import ftw_vision_name

    with pytest.raises(ValueError):
        ftw_vision_name("model.language_model.layers.0.self_attn.q_proj.weight")


def test_mtp_still_classifies_as_skip():
    """Sanity: removing vision from SKIP_PREFIXES must not have disturbed MTP's own
    defense-in-depth skip (it still has its own dedicated pass-through, separate kind)."""
    from freetoken.checkpoint.qwen_layout import DEST_SKIP, classify_tensor

    assert classify_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight") == DEST_SKIP
    assert classify_tensor("mtp.fc_embedding.weight") == DEST_SKIP


# ---------------------------------------------------------------------------
# Slower, real-data test: convert.py's actual generator, end to end.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_iter_vision_from_disk_yields_carried_reshaped_tensors(real_vision_keys):
    """Exercises convert.py's real ``_iter_vision_from_disk`` (no FTWWriter, no disk
    write) -- the exact generator the FTW converter's dedicated vision pass-through
    consumes. Confirms: all 333 names, all renamed (``vision_tower.`` prefix), and the
    Conv3d -> Linear reshape landed at ``[1152, 1536]`` (fact 3 in the assignment)."""
    from freetoken.checkpoint.convert import _iter_vision_from_disk

    seen: dict[str, tuple[int, ...]] = {}
    for name, tensor in _iter_vision_from_disk(MODEL_PATH, dtype=torch.bfloat16):
        assert name.startswith("vision_tower.")
        seen[name] = tuple(tensor.shape)

    assert len(seen) == 333, f"expected 333 carried vision tensors, got {len(seen)}"
    expected_names = {"vision_tower." + k[len("model.visual."):] for k in real_vision_keys}
    assert set(seen) == expected_names

    patch_embed_shape = seen["vision_tower.patch_embed.proj.weight"]
    assert patch_embed_shape == (1152, 1536), (
        f"patch_embed.proj.weight: expected reshaped Linear-equivalent [1152, 1536], "
        f"got {patch_embed_shape}"
    )
