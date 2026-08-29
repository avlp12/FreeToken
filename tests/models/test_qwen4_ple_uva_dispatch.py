"""FREETOKEN_PLE_UVA=1 wiring: the new pinned-host/UVA gather backend must be
opt-in, cached, fail safely back to the existing (unmodified) mmap gather, and
never even be considered off a CUDA device -- all without ever constructing a
real pinned allocation or launching the Triton kernel (see
tests/kernels/test_ple_iq4nl_gather.py for the dequant-formula proof and the
GPU-only kernel-launch test).

Background: ``_NGramTable.gather()`` (freetoken/models/qwen4_exp/ngram.py) took a
new branch ahead of its pre-existing mmap path (host sync + python shard loop +
unpinned H2D -- see that method's and precompute_decode_ngram's docstrings for why
that's the ~6.4% decode-step cost this branch exists to remove). The branch only
fires when FREETOKEN_PLE_UVA=1, the table is the GGUF-sourced IQ4_NL/FTW one, ids
are on a CUDA device, AND backend construction has not already failed once for
this table instance (``_NGramTable._uva_backend_for`` caches both outcomes). This
file exercises that gating and caching logic with fakes standing in for
``_PLEUVABackend``, so it needs no GPU, no real checkpoint, and never touches
``freetoken.kernel.pinned`` / ``freetoken.kernel.triton.ple`` for real.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import freetoken.models.qwen4_exp.ngram as ngram
from freetoken.models.qwen4_exp.ngram import _NGramTable

HEAD_DIM = 160


def _fake_table(*, iq4nl: bool = True, ftw: bool = True) -> _NGramTable:
    """A ``_NGramTable`` with just enough attributes for ``gather()`` /
    ``_uva_backend_for`` to run, none of it backed by real checkpoint files."""
    table = _NGramTable.__new__(_NGramTable)
    table.key_base = "dummy.ple_embedding.ngram_embedding"
    table._iq4nl = iq4nl
    table._ftw_locs = {"dummy": None} if ftw else None
    table.rows_per_shard = 100
    table.split_parts = 2
    table.head_dim = HEAD_DIM
    table.weight_scale = 1.0
    table._uva_backend = None
    table._uva_failed = False

    def fake_rows(shard_i: int, local: torch.Tensor) -> torch.Tensor:
        # Deterministic, cheap stand-in for the real mmap dequantized rows: encode
        # (shard_i, local row) into the output so a dispatch bug that reads the
        # wrong shard/row would show up as a value mismatch, not just a shape match.
        base = shard_i * 1000
        return (base + local).float().unsqueeze(-1).expand(-1, HEAD_DIM).clone()

    table._rows = fake_rows
    return table


@pytest.fixture(autouse=True)
def _restore_ple_uva_flag():
    saved = ngram._PLE_UVA
    yield
    ngram._PLE_UVA = saved


def test_uva_backend_for_builds_once_and_caches_the_instance(monkeypatch):
    table = _fake_table()
    calls = []

    class FakeBackend:
        def __init__(self, tbl, device):
            calls.append((tbl, device))

    monkeypatch.setattr(ngram, "_PLEUVABackend", FakeBackend)

    device = torch.device("cpu")  # _uva_backend_for itself doesn't gate on device type
    first = table._uva_backend_for(device)
    second = table._uva_backend_for(device)

    assert isinstance(first, FakeBackend)
    assert first is second
    assert len(calls) == 1, "backend must be constructed once and cached, not per-call"


def test_uva_backend_for_caches_failure_and_never_retries(monkeypatch):
    table = _fake_table()
    attempts = []

    class ExplodingBackend:
        def __init__(self, tbl, device):
            attempts.append(1)
            raise RuntimeError("simulated: pin budget exceeded")

    monkeypatch.setattr(ngram, "_PLEUVABackend", ExplodingBackend)

    device = torch.device("cpu")
    assert table._uva_backend_for(device) is None
    assert table._uva_backend_for(device) is None
    assert len(attempts) == 1, (
        "a failed backend construction must be cached (self._uva_failed) so gather() "
        "falls back to mmap on every subsequent call, not just the first"
    )


def test_gather_never_builds_the_uva_backend_on_a_cpu_device(monkeypatch):
    """Core safety property: even with FREETOKEN_PLE_UVA=1 and an IQ4_NL/FTW table,
    a CPU-device gather() call must take the unmodified mmap path and must never
    even attempt to build the UVA backend (which would try to pin host memory --
    illegal/pointless off a CUDA device, and exactly the kind of accidental-GPU-
    touch this project's operating rules forbid triggering from a CPU-only test)."""
    ngram._PLE_UVA = True
    table = _fake_table(iq4nl=True, ftw=True)

    def explode(device):
        raise AssertionError("_uva_backend_for must not be called for a CPU gather")

    monkeypatch.setattr(table, "_uva_backend_for", explode)

    ids = torch.tensor([[5, 105]], dtype=torch.int64)  # shard 0 row 5, shard 1 row 5
    out = table.gather(ids, torch.float32)

    assert out.shape == (1, 2, HEAD_DIM)
    torch.testing.assert_close(out[0, 0], torch.full((HEAD_DIM,), 5.0))
    torch.testing.assert_close(out[0, 1], torch.full((HEAD_DIM,), 1005.0))


def test_gather_with_flag_unset_matches_gather_with_flag_set_off_cpu(monkeypatch):
    """FREETOKEN_PLE_UVA=1 must be a pure no-op for a CPU-device caller: same
    branch, same result, whether or not the flag is set (the flag only matters once
    a CUDA device is in play -- see the test above)."""
    table_a = _fake_table()
    table_b = _fake_table()
    ids = torch.tensor([3, 50, 199], dtype=torch.int64)

    ngram._PLE_UVA = False
    off = table_a.gather(ids, torch.float32)
    ngram._PLE_UVA = True
    on = table_b.gather(ids, torch.float32)

    torch.testing.assert_close(off, on)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device tensor")
def test_gather_dispatches_to_the_uva_backend_on_a_cuda_device(monkeypatch):
    """GPU-adjacent (but not GPU-heavy): a fake backend stands in for
    ``_PLEUVABackend`` so this only proves gather() calls it and reshapes its
    output correctly -- it does not exercise the real Triton kernel or pinned
    memory (see tests/kernels/test_ple_iq4nl_gather.py for that). Skipped in a
    CPU-only run; advisor should confirm this passes on the GPU box."""
    ngram._PLE_UVA = True
    table = _fake_table()
    device = torch.device("cuda", torch.cuda.current_device())

    class FakeBackend:
        def __init__(self, tbl, dev):
            self.calls = []

        def gather(self, ids_flat, out_dtype):
            self.calls.append(ids_flat.clone())
            return (ids_flat.float().unsqueeze(-1).expand(-1, HEAD_DIM)).to(out_dtype)

    fake = FakeBackend(table, device)
    monkeypatch.setattr(table, "_uva_backend_for", lambda dev: fake)

    ids = torch.tensor([[7, 8]], dtype=torch.int64, device=device)
    out = table.gather(ids, torch.bfloat16)

    assert out.shape == (1, 2, HEAD_DIM)
    assert out.device.type == "cuda"
    torch.testing.assert_close(out[0, 0].float().cpu(), torch.full((HEAD_DIM,), 7.0))
    torch.testing.assert_close(out[0, 1].float().cpu(), torch.full((HEAD_DIM,), 8.0))


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("ple_uva_env", [None, "1"])
def test_import_freetoken_needs_no_gpu(ple_uva_env):
    """Acceptance criterion: `python -c "import freetoken"` must not break and must
    not need a GPU, whether or not FREETOKEN_PLE_UVA is set (setting the flag only
    flips a module-level bool read at import time -- it must not eagerly import
    triton or freetoken.kernel._pinned_tensor). Runs in a fresh subprocess (see
    tests/daemon/test_daemon_import_safety.py for why: a same-process check would
    diff against a sys.modules baseline already polluted by other tests' imports)
    with CUDA_VISIBLE_DEVICES="" so a real GPU being present can't mask a bug that
    would surface on a CPU-only box."""
    pkg = os.path.join(_REPO_ROOT, "python")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pkg + (os.pathsep + existing if existing else "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    if ple_uva_env is None:
        env.pop("FREETOKEN_PLE_UVA", None)
    else:
        env["FREETOKEN_PLE_UVA"] = ple_uva_env

    proc = subprocess.run(
        [sys.executable, "-c", "import freetoken; print('import freetoken OK')"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"import freetoken failed (FREETOKEN_PLE_UVA={ple_uva_env!r})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "import freetoken OK" in proc.stdout
