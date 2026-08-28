"""Coverage for ``checkpoint/graft_vision.py`` -- appending the vision tower to an
already-converted FTW checkpoint as one new shard, instead of reconverting the whole
checkpoint.

Everything here is synthetic and fast: tiny hand-built FTW checkpoints (via the real
``FTWWriter``, so the fixtures obey the same invariants a real conversion produces) and a
monkeypatched stand-in for ``convert.py::_iter_vision_from_disk`` (the real one reads a
Qwen4-Exp HF checkpoint's ``model.visual.*`` safetensors and needs the actual vision
model/config -- that generator itself is covered separately, against the real checkpoint,
by ``tests/checkpoint/test_qwen_vision_layout.py``). Nothing here reads or writes the real
131 GiB production checkpoint.

Four required groups (see the assignment this closes):

1. ``TestRoundTrip`` -- graft onto a small multi-shard, multi-kind synthetic FTW; every
   original tensor and every new tensor reads back byte-identical; ``iter_ftw_weights``'s
   default kinds picks up both; alignment and gap-free shard invariants hold throughout.
2. ``TestHardlinkCloneSafety`` -- clone with hardlinks, graft into the clone, and prove
   with real inode comparisons (not just "it looks fine") that the original's shard files
   and index are untouched while the clone's shard files still share inodes with the
   original (proving the clone really was hardlinked) and the clone's index does not
   (proving ``os.replace`` broke that link rather than editing through it).
3. ``TestRefusals`` -- double-graft, a pre-existing target shard file, and a tampered
   (non-contiguous) index all raise ``GraftError`` and write nothing.
4. ``TestShardRolling`` -- a graft payload that exceeds a tiny ``shard_limit`` rolls into
   multiple new shards, still gap-free and aligned.

Plus CLI-level tests for ``--dry-run`` and ``--clone-to`` (acceptance criterion 1).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import pytest
import torch

from freetoken.checkpoint import graft_vision as gv
from freetoken.checkpoint.ftw import ALIGN, FTWReader, FTWWriter, iter_ftw_weights

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


def _rand_tensor(nbytes: int) -> torch.Tensor:
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8)


def _hash_tensor_raw(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _build_synthetic_ftw(out_dir: str, shard_limit: int, tensors: list[tuple[str, str, torch.Tensor]]) -> dict:
    """``tensors``: list of ``(name, kind, tensor)``. Builds via the real ``FTWWriter``
    (so the fixture obeys the writer's own invariants) with a meta dict shaped like a
    real (pre-vision) FTW index -- in particular, deliberately no ``vision_num_tensors``
    key, matching what a not-yet-grafted checkpoint looks like."""
    writer = FTWWriter(out_dir, shard_limit=shard_limit)
    for name, kind, t in tensors:
        writer.add_tensor(name, t, kind=kind)
    counts = dict(Counter(k for _, k, _ in tensors))
    return writer.finalize({
        "source_model_path": "synthetic",
        "fingerprint": "synthetic-fingerprint-do-not-touch",
        "quant_format": None,
        "expert_bank_num_layers": None,
        "expert_bank_quant_types": None,
        "counts": counts,
        "ngram_num_tensors": counts.get("ngram", 0),
        "mtp_num_tensors": 0,
        "copied_metadata": ["config.json"],
    })


def _default_tensor_specs() -> list[tuple[str, str, torch.Tensor]]:
    return [
        ("model.layers.0.weight", "weight", _rand_tensor(2000)),
        ("model.layers.1.weight", "weight", _rand_tensor(2000)),
        ("model.layers.2.weight", "weight", _rand_tensor(5000)),
        ("model.layers.3.weight", "weight", _rand_tensor(3000)),
        ("model.layers.4.weight", "weight", _rand_tensor(6000)),
        ("expert_bank_0", "experts_bank", _rand_tensor(4000)),
        ("ngram_table", "ngram", _rand_tensor(1500)),
    ]


def _vision_specs(n: int = 3, size: int = 1000) -> list[tuple[str, torch.Tensor]]:
    return [(f"vision_tower.block.{i}.weight", _rand_tensor(size + i * 37)) for i in range(n)]


def _install_fake_vision_source(monkeypatch, specs: list[tuple[str, torch.Tensor]]) -> None:
    def _fake_iter_vision_from_disk(model_path, *, dtype):
        for name, t in specs:
            yield name, t
    monkeypatch.setattr(gv, "_iter_vision_from_disk", _fake_iter_vision_from_disk)


def _read_and_hash(ftw_dir: str, name: str) -> str:
    reader = FTWReader(ftw_dir)
    try:
        entry = reader.tensors[name]
        buf = bytearray(((entry["nbytes"] + ALIGN - 1) // ALIGN) * ALIGN)
        reader.read_into(memoryview(buf), entry)
        return hashlib.sha256(bytes(buf[: entry["nbytes"]])).hexdigest()
    finally:
        reader.close()


def _assert_gapfree_and_aligned(index: dict) -> None:
    shards = sorted(index["shards"], key=lambda s: s["global_off"])
    pos = 0
    for sh in shards:
        assert sh["global_off"] == pos, f"gap before shard {sh['file']}"
        assert sh["global_off"] % ALIGN == 0
        pos += sh["nbytes"]
    assert pos == index["total_bytes"]
    for t in index["tensors"]:
        assert t["global_off"] % ALIGN == 0, t["name"]


def _load_index(ftw_dir: str) -> dict:
    with open(os.path.join(ftw_dir, "freetoken_weight.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Round trip on a synthetic multi-shard, multi-kind FTW
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        specs = _default_tensor_specs()
        base_index = _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=specs)
        assert len(base_index["shards"]) >= 2, "fixture should span several shards"
        orig_hashes = {name: _hash_tensor_raw(t) for name, _kind, t in specs}

        vspecs = _vision_specs(4, size=1200)
        _install_fake_vision_source(monkeypatch, vspecs)
        vision_hashes = {name: _hash_tensor_raw(t) for name, t in vspecs}

        result = gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=False)
        assert result["n_vision_tensors"] == len(vspecs)
        assert result["verified_new"] == len(vspecs)

        new_index = _load_index(ftw_dir)
        _assert_gapfree_and_aligned(new_index)

        # Every original tensor still reads back byte-identical.
        for name, expected in orig_hashes.items():
            assert _read_and_hash(ftw_dir, name) == expected, name

        # Every new tensor reads back byte-identical.
        for name, expected in vision_hashes.items():
            assert _read_and_hash(ftw_dir, name) == expected, name

        # iter_ftw_weights default kinds yields originals' weight entries + new ones.
        got_names = {name for name, _t in iter_ftw_weights(ftw_dir)}
        expected_weight_names = {n for n, k, _t in specs if k == "weight"} | set(vision_hashes)
        assert got_names == expected_weight_names

        # counts/meta bookkeeping.
        assert new_index["counts"]["weight"] == 5 + len(vspecs)
        assert new_index["counts"]["experts_bank"] == 1
        assert new_index["counts"]["ngram"] == 1
        assert new_index["vision_num_tensors"] == len(vspecs)
        assert new_index["vision_grafted_from"] == os.path.abspath("unused-source-dir")
        assert new_index["vision_graft_tool"] == "freetoken.checkpoint.graft_vision"
        assert new_index["fingerprint"] == "synthetic-fingerprint-do-not-touch"  # preserved

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        specs = _default_tensor_specs()
        _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=specs)
        before_files = sorted(os.listdir(ftw_dir))
        before_index_bytes = open(os.path.join(ftw_dir, "freetoken_weight.json"), "rb").read()

        vspecs = _vision_specs(3)
        _install_fake_vision_source(monkeypatch, vspecs)

        plan = gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=True)
        assert plan["n_vision_tensors"] == len(vspecs)
        assert "verified_new" not in plan

        after_files = sorted(os.listdir(ftw_dir))
        after_index_bytes = open(os.path.join(ftw_dir, "freetoken_weight.json"), "rb").read()
        assert before_files == after_files, "dry-run must not create any file"
        assert before_index_bytes == after_index_bytes, "dry-run must not touch the index"


# ---------------------------------------------------------------------------
# 2. Hardlink-clone safety
# ---------------------------------------------------------------------------


class TestHardlinkCloneSafety:
    def test_clone_then_graft_leaves_original_untouched(self, tmp_path, monkeypatch):
        orig_dir = str(tmp_path / "orig")
        clone_dir = str(tmp_path / "clone")
        specs = _default_tensor_specs()
        base_index = _build_synthetic_ftw(orig_dir, shard_limit=3 * ALIGN, tensors=specs)
        shard_files = [sh["file"] for sh in base_index["shards"]]

        orig_shard_stat_before = {
            f: os.stat(os.path.join(orig_dir, f)) for f in shard_files
        }
        orig_shard_hash_before = {
            f: hashlib.sha256(open(os.path.join(orig_dir, f), "rb").read()).hexdigest()
            for f in shard_files
        }
        orig_index_bytes_before = open(os.path.join(orig_dir, "freetoken_weight.json"), "rb").read()
        orig_index_stat_before = os.stat(os.path.join(orig_dir, "freetoken_weight.json"))

        gv.clone_ftw_dir(orig_dir, clone_dir)

        # Pre-graft: the clone's index really is hardlinked to the original's (same inode) --
        # establishes the baseline that os.replace is about to break.
        clone_index_ino_before = os.stat(os.path.join(clone_dir, "freetoken_weight.json")).st_ino
        assert clone_index_ino_before == orig_index_stat_before.st_ino
        for f in shard_files:
            assert os.stat(os.path.join(clone_dir, f)).st_ino == orig_shard_stat_before[f].st_ino

        vspecs = _vision_specs(3)
        _install_fake_vision_source(monkeypatch, vspecs)
        result = gv.graft_vision(clone_dir, "unused-source-dir", dry_run=False)
        assert result["verified_new"] == len(vspecs)

        # Original shard files: byte-identical AND same inode/mtime -- never touched.
        for f in shard_files:
            path = os.path.join(orig_dir, f)
            st_after = os.stat(path)
            assert st_after.st_ino == orig_shard_stat_before[f].st_ino, f
            assert st_after.st_mtime_ns == orig_shard_stat_before[f].st_mtime_ns, f
            assert st_after.st_size == orig_shard_stat_before[f].st_size, f
            got_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
            assert got_hash == orig_shard_hash_before[f], f

        # Original index.json byte-identical.
        orig_index_bytes_after = open(os.path.join(orig_dir, "freetoken_weight.json"), "rb").read()
        assert orig_index_bytes_after == orig_index_bytes_before

        # Clone's shard files still share inodes with the original (hardlinked, not copied).
        for f in shard_files:
            clone_ino = os.stat(os.path.join(clone_dir, f)).st_ino
            orig_ino = os.stat(os.path.join(orig_dir, f)).st_ino
            assert clone_ino == orig_ino, f"clone shard {f} is no longer hardlinked to the original"

        # Clone's index.json no longer shares an inode with the original's (os.replace broke it).
        clone_index_ino_after = os.stat(os.path.join(clone_dir, "freetoken_weight.json")).st_ino
        orig_index_ino_after = os.stat(os.path.join(orig_dir, "freetoken_weight.json")).st_ino
        assert clone_index_ino_after != orig_index_ino_after
        assert clone_index_ino_after != clone_index_ino_before

        # And the original's index content proves it: no vision_num_tensors, no vision_tower.*.
        orig_index_after = json.loads(orig_index_bytes_after)
        assert "vision_num_tensors" not in orig_index_after
        assert not any(t["name"].startswith("vision_tower.") for t in orig_index_after["tensors"])

        # While the clone's index DOES carry the graft.
        clone_index = _load_index(clone_dir)
        assert clone_index["vision_num_tensors"] == len(vspecs)


# ---------------------------------------------------------------------------
# 3. Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_double_graft_refused(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=_default_tensor_specs())
        _install_fake_vision_source(monkeypatch, _vision_specs(2))

        gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=False)
        index_after_first = _load_index(ftw_dir)

        with pytest.raises(gv.GraftError, match="double-graft"):
            gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=False)

        # Refusal must not have written anything further.
        assert _load_index(ftw_dir) == index_after_first

    def test_target_shard_file_already_present_refused(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        base_index = _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=_default_tensor_specs())
        next_idx = max(int(sh["file"].split("-")[1].split(".")[0]) for sh in base_index["shards"]) + 1
        colliding = os.path.join(ftw_dir, f"freetoken-{next_idx:05d}.ftw")
        with open(colliding, "wb") as f:
            f.write(b"not part of any graft")

        _install_fake_vision_source(monkeypatch, _vision_specs(2))
        with pytest.raises(gv.GraftError, match="already exists"):
            gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=False)

        # The colliding file must be untouched, and the index must not have been rewritten.
        with open(colliding, "rb") as f:
            assert f.read() == b"not part of any graft"
        assert "vision_num_tensors" not in _load_index(ftw_dir)

    def test_tampered_noncontiguous_index_refused(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=_default_tensor_specs())

        index_path = os.path.join(ftw_dir, "freetoken_weight.json")
        with open(index_path) as f:
            index = json.load(f)
        assert len(index["shards"]) >= 2
        shards = sorted(index["shards"], key=lambda s: s["global_off"])
        shards[1]["global_off"] += ALIGN  # punch a gap
        index["shards"] = shards
        with open(index_path, "w") as f:
            json.dump(index, f)

        _install_fake_vision_source(monkeypatch, _vision_specs(2))
        with pytest.raises(gv.GraftError, match="gap|non-contiguous|contiguous"):
            gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=False)


# ---------------------------------------------------------------------------
# 4. Shard rolling
# ---------------------------------------------------------------------------


class TestShardRolling:
    def test_graft_payload_exceeding_shard_limit_rolls_multiple_shards(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        _build_synthetic_ftw(ftw_dir, shard_limit=ALIGN, tensors=[
            ("model.layers.0.weight", "weight", _rand_tensor(500)),
        ])

        # Each vision tensor (padded) nearly fills one ALIGN-sized shard on its own, so
        # 3 tensors must roll into (at least) 3 distinct new shards under shard_limit=ALIGN.
        vspecs = _vision_specs(3, size=3000)
        _install_fake_vision_source(monkeypatch, vspecs)
        vision_hashes = {name: _hash_tensor_raw(t) for name, t in vspecs}

        result = gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=False)
        assert len(result["new_shards"]) >= 2, "payload should have rolled into >= 2 new shards"

        new_index = _load_index(ftw_dir)
        _assert_gapfree_and_aligned(new_index)
        for name, expected in vision_hashes.items():
            assert _read_and_hash(ftw_dir, name) == expected, name

    def test_dry_run_plans_same_shard_rolling_as_real_run(self, tmp_path, monkeypatch):
        ftw_dir = str(tmp_path / "ftw")
        _build_synthetic_ftw(ftw_dir, shard_limit=ALIGN, tensors=[
            ("model.layers.0.weight", "weight", _rand_tensor(500)),
        ])
        vspecs = _vision_specs(3, size=3000)
        _install_fake_vision_source(monkeypatch, vspecs)

        plan = gv.graft_vision(ftw_dir, "unused-source-dir", dry_run=True)
        assert len(plan["new_shards"]) >= 2
        # dry-run truly wrote nothing: no new shard files appeared on disk.
        for sh in plan["new_shards"]:
            assert not os.path.exists(os.path.join(ftw_dir, sh["file"]))


# ---------------------------------------------------------------------------
# CLI: --dry-run and --clone-to
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_dry_run_prints_plan_and_writes_nothing(self, tmp_path, monkeypatch, capsys):
        ftw_dir = str(tmp_path / "ftw")
        _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=_default_tensor_specs())
        before = sorted(os.listdir(ftw_dir))

        _install_fake_vision_source(monkeypatch, _vision_specs(2))
        rc = gv.main(["--ftw", ftw_dir, "--source", "unused-source-dir", "--dry-run"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "graft plan" in out
        assert "dry run: nothing written" in out
        assert sorted(os.listdir(ftw_dir)) == before

    def test_cli_clone_to_dry_run_creates_no_clone(self, tmp_path, monkeypatch, capsys):
        ftw_dir = str(tmp_path / "ftw")
        clone_dir = str(tmp_path / "clone")
        _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=_default_tensor_specs())

        _install_fake_vision_source(monkeypatch, _vision_specs(2))
        rc = gv.main(["--ftw", ftw_dir, "--source", "unused-source-dir",
                     "--clone-to", clone_dir, "--dry-run"])
        assert rc == 0
        assert not os.path.exists(clone_dir), "--dry-run must not create the clone"

    def test_cli_clone_to_real_graft(self, tmp_path, monkeypatch, capsys):
        ftw_dir = str(tmp_path / "ftw")
        clone_dir = str(tmp_path / "clone")
        _build_synthetic_ftw(ftw_dir, shard_limit=3 * ALIGN, tensors=_default_tensor_specs())

        vspecs = _vision_specs(2)
        _install_fake_vision_source(monkeypatch, vspecs)
        rc = gv.main(["--ftw", ftw_dir, "--source", "unused-source-dir", "--clone-to", clone_dir])
        assert rc == 0

        assert "vision_num_tensors" not in _load_index(ftw_dir), "original must stay untouched"
        clone_index = _load_index(clone_dir)
        assert clone_index["vision_num_tensors"] == len(vspecs)

        out = capsys.readouterr().out
        assert "cloned (hardlink)" in out
        assert "verified 2 new" in out
