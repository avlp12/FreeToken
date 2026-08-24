"""``iter_gguf_tensors`` must read a llama.cpp split-GGUF shard set (a directory of
``*-00001-of-0000N.gguf`` siblings, or the first shard's path directly) and yield the
union of tensors across shards -- each shard's tensor-info table only lists the
tensors physically stored in that shard.

Builds tiny synthetic shards with the ``gguf`` pip package's own writer (so this
does not depend on any real checkpoint, complete or not) and cross-checks against a
single-file GGUF holding the identical tensors, to pin down "split reads must match
single-file reads of the same data."
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from freetoken.models.gguf.reader import is_gguf_path, iter_gguf_tensors  # noqa: E402

# name -> (numpy array, ggml type to store as)
_TENSORS = {
    "blk.0.attn_norm.weight": (np.arange(32, dtype=np.float32).reshape(1, 32), "F16"),
    "blk.0.ffn_gate.weight": (
        np.linspace(-1, 1, 2 * 32, dtype=np.float32).reshape(2, 32),
        "Q4_0",
    ),
    "blk.1.attn_norm.weight": (np.arange(32, 64, dtype=np.float32).reshape(1, 32), "F16"),
    "blk.1.ffn_gate.weight": (
        np.linspace(1, -1, 3 * 32, dtype=np.float32).reshape(3, 32),
        "Q4_0",
    ),
}


def _write_gguf(path: str, split_max_tensors: int) -> None:
    from gguf.quants import quantize

    writer = gguf.GGUFWriter(path=path, arch="llama", split_max_tensors=split_max_tensors)
    for name, (arr, kind) in _TENSORS.items():
        if kind == "F16":
            writer.add_tensor(name, arr.astype(np.float16))
        else:
            qtype = gguf.GGMLQuantizationType.Q4_0
            writer.add_tensor(name, quantize(arr, qtype), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture()
def split_gguf_dir(tmp_path):
    """A directory holding a 2-shard split-GGUF set (2 tensors per shard)."""
    out = str(tmp_path / "synth" / "synth-model")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_gguf(out, split_max_tensors=2)
    shards = sorted(glob.glob(os.path.dirname(out) + "/*.gguf"))
    assert len(shards) == 2, f"expected 2 shards, got {shards}"
    assert shards[0].endswith("-00001-of-00002.gguf")
    assert shards[1].endswith("-00002-of-00002.gguf")
    return os.path.dirname(out), shards


@pytest.fixture()
def single_gguf_file(tmp_path):
    """The identical tensor set written as one plain (non-split) GGUF file."""
    out = str(tmp_path / "single.gguf")
    _write_gguf(out, split_max_tensors=0)
    assert os.path.isfile(out)
    return out


def _by_name(tensors):
    return {t.name: t for t in tensors}


def test_split_dir_and_first_shard_are_detected_as_gguf(split_gguf_dir):
    dirpath, shards = split_gguf_dir
    assert is_gguf_path(dirpath)
    assert is_gguf_path(shards[0])


def test_split_dir_yields_union_of_tensors_across_shards(split_gguf_dir):
    dirpath, _ = split_gguf_dir
    got = _by_name(iter_gguf_tensors(dirpath))
    assert set(got) == set(_TENSORS)


def test_first_shard_path_yields_same_tensors_as_directory(split_gguf_dir):
    dirpath, shards = split_gguf_dir
    from_dir = _by_name(iter_gguf_tensors(dirpath))
    from_first_shard = _by_name(iter_gguf_tensors(shards[0]))
    assert set(from_dir) == set(from_first_shard) == set(_TENSORS)


def test_split_read_matches_single_file_read_of_identical_data(split_gguf_dir, single_gguf_file):
    dirpath, _ = split_gguf_dir
    split_tensors = _by_name(iter_gguf_tensors(dirpath))
    single_tensors = _by_name(iter_gguf_tensors(single_gguf_file))
    assert set(split_tensors) == set(single_tensors) == set(_TENSORS)

    for name in _TENSORS:
        a, b = split_tensors[name], single_tensors[name]
        assert a.shape == b.shape
        assert a.ggml_type == b.ggml_type
        assert a.rows == b.rows
        assert a.row_bytes == b.row_bytes
        torch_a, torch_b = a.packed(), b.packed()
        assert torch_a.shape == torch_b.shape
        assert (torch_a == torch_b).all()


def test_split_tensor_shapes_and_types_match_source_arrays(split_gguf_dir):
    dirpath, _ = split_gguf_dir
    got = _by_name(iter_gguf_tensors(dirpath))
    for name, (arr, kind) in _TENSORS.items():
        t = got[name]
        assert t.shape == arr.shape
        expected_type = gguf.GGMLQuantizationType.F16 if kind == "F16" else gguf.GGMLQuantizationType.Q4_0
        assert t.ggml_type == int(expected_type)
