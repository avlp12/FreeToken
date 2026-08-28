"""Graft the vision tower onto an existing FTW checkpoint WITHOUT reconverting it.

    python -m freetoken.checkpoint.graft_vision \
        --ftw <ftw_dir> --source <hf_dir_with_model.visual.*> \
        [--clone-to <new_dir>] [--dtype bfloat16] [--dry-run]

Why this exists: adding the vision tower to an already-converted FTW checkpoint by
reconverting means rewriting the whole checkpoint (100+ GiB, hours, disk you may not
have free) to add ~0.8 GiB of new tensors. The FTW format's logical byte stream is
append-friendly -- see ``checkpoint/ftw.py``'s module docstring -- so this tool instead
appends ONE new shard holding just the vision tensors and extends the index in place.

Reuses ``checkpoint/convert.py::_iter_vision_from_disk`` for the vision tensor
source (naming/reshape rule lives there exactly once) and ``checkpoint/ftw.py``'s own
alignment/shard-naming helpers (``ALIGN``, ``_SHARD_FMT``, ``_align_up``, ``_dtype_str``)
rather than re-deriving them, so this module can never drift out of sync with
``FTWWriter``'s invariants.

Safety model (this is the load-bearing part):

* ``--clone-to`` makes a HARDLINK clone of ``--ftw`` (``os.link`` per file -- the
  ``cp -al`` effect): the clone's shard files and index.json are the SAME INODES as the
  original's. Grafting into the clone must never write through one of those shared
  inodes, or the "clone" would silently corrupt the production checkpoint it was
  supposed to leave untouched.
* :class:`_GraftWriter` enforces this for the shard files: it NEVER opens an existing
  ``.ftw`` file for writing. It always starts a brand-new shard at the next unused
  shard index (refusing loudly if that filename already exists), even though the
  checkpoint's last existing shard may have spare room under ``shard_limit`` -- that
  spare room is deliberately left unused rather than risk a write into a shared inode.
* The index is rewritten the same way :meth:`FTWWriter.finalize` does: build the new
  dict in memory, write it to a fresh temp file, then ``os.replace`` the temp file onto
  ``freetoken_weight.json``. ``os.replace`` is a rename, which repoints the directory
  entry at a new inode -- it does not mutate the old inode's bytes, so any OTHER
  directory entry still linked to that old inode (the original checkpoint's own
  ``freetoken_weight.json``, before the clone's copy was replaced) is unaffected. This
  is exactly how ``cp -al`` + editing a file "the normal way" (not in place) is safe.

Everything the tool writes is verified before it reports success: the new tensors are
re-read through a fresh :class:`~freetoken.checkpoint.ftw.FTWReader` and hash-compared
against what was written, and a sample of pre-existing tensors is hash-compared against
what it read BEFORE the graft, proving the graft did not disturb the rest of the
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time

import torch

from freetoken.utils import init_logger

from .convert import _iter_vision_from_disk
from .ftw import ALIGN, FORMAT_TAG, INDEX_NAME, _SHARD_FMT, _align_up, _dtype_str, FTWReader

logger = init_logger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

VISION_PREFIX = "vision_tower."
_SHARD_RE = re.compile(r"^freetoken-(\d{5})\.ftw$")
_DEFAULT_VERIFY_SAMPLE = 8


class GraftError(RuntimeError):
    """A graft precondition failed, or a post-write self-check failed. Always raised
    BEFORE (preconditions) or AFTER (self-verify) any bytes are written to a *new*
    shard file -- never leaves a half-graft: the index is only rewritten once, at the
    very end, via :func:`_write_index`'s temp-file + ``os.replace``."""


# ============================== index helpers ==============================


def _load_index(ftw_dir: str) -> dict:
    idx_path = os.path.join(ftw_dir, INDEX_NAME)
    if not os.path.isfile(idx_path):
        raise GraftError(f"not an FTW checkpoint (missing {INDEX_NAME}): {ftw_dir}")
    with open(idx_path) as f:
        return json.load(f)


def _validate_index(index: dict, ftw_dir: str) -> int:
    """Refuse loudly on anything :class:`_GraftWriter` would otherwise trust blindly.
    Returns the validated ``shard_limit``."""
    if index.get("format") != FORMAT_TAG:
        raise GraftError(f"{ftw_dir}: index format {index.get('format')!r} != {FORMAT_TAG!r}")

    shard_limit = index.get("shard_limit")
    if not isinstance(shard_limit, int) or shard_limit <= 0 or shard_limit % ALIGN != 0:
        raise GraftError(f"{ftw_dir}: shard_limit {shard_limit!r} missing or not a positive "
                         f"multiple of {ALIGN}")

    total_bytes = index.get("total_bytes")
    if not isinstance(total_bytes, int) or total_bytes % ALIGN != 0:
        raise GraftError(f"{ftw_dir}: total_bytes {total_bytes!r} missing or not "
                         f"{ALIGN}-aligned")

    shards = sorted(index.get("shards", []), key=lambda s: s["global_off"])
    if not shards:
        raise GraftError(f"{ftw_dir}: index has no shards")
    pos = 0
    for sh in shards:
        if sh["global_off"] != pos:
            raise GraftError(
                f"{ftw_dir}: shard {sh['file']!r} starts at {sh['global_off']}, expected "
                f"{pos} -- non-contiguous/gapped shard list (tampered or corrupt index)"
            )
        pos += sh["nbytes"]
    if pos != total_bytes:
        raise GraftError(
            f"{ftw_dir}: shards sum to {pos} bytes but index total_bytes={total_bytes}"
        )

    if "vision_num_tensors" in index:
        raise GraftError(
            f"{ftw_dir}: index already has vision_num_tensors="
            f"{index['vision_num_tensors']!r} -- refusing to double-graft"
        )
    existing_vision = [t["name"] for t in index.get("tensors", []) if t["name"].startswith(VISION_PREFIX)]
    if existing_vision:
        raise GraftError(
            f"{ftw_dir}: index already has {len(existing_vision)} {VISION_PREFIX}* "
            f"entries (e.g. {existing_vision[0]!r}) -- refusing to double-graft"
        )
    return shard_limit


def _next_shard_idx(shards: list[dict]) -> int:
    max_idx = -1
    for sh in shards:
        m = _SHARD_RE.match(sh["file"])
        if not m:
            raise GraftError(f"unrecognized shard filename in index: {sh['file']!r}")
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _check_target_shard_absent(ftw_dir: str, shard_idx: int) -> None:
    path = os.path.join(ftw_dir, _SHARD_FMT.format(shard_idx))
    if os.path.exists(path):
        raise GraftError(
            f"refusing to graft: target shard file already exists: {path} "
            "(a graft never opens an existing .ftw file for writing -- if this dir was "
            "already grafted or partially grafted, investigate before retrying)"
        )


def _write_index(ftw_dir: str, index: dict) -> None:
    """Temp file + ``os.replace``, exactly like ``FTWWriter.finalize`` -- see the module
    docstring for why this is what keeps a hardlinked clone's original index safe."""
    tmp = os.path.join(ftw_dir, INDEX_NAME + ".graft.tmp")
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, os.path.join(ftw_dir, INDEX_NAME))


def _stat_shards(ftw_dir: str, shards: list[dict]) -> dict[str, tuple]:
    out = {}
    for sh in shards:
        st = os.stat(os.path.join(ftw_dir, sh["file"]))
        out[sh["file"]] = (st.st_ino, st.st_size, st.st_mtime_ns)
    return out


# ============================== writer ==============================


class _GraftWriter:
    """Continues an existing FTW's logical byte stream with brand-new shard files.

    Mirrors ``FTWWriter._roll``/``add_tensor``/``finalize``'s invariants (4096-aligned
    tensor starts, per-tensor padding, ``shard_limit`` rolling, a small tensor never
    splits across shards) but NEVER opens an existing shard file: every shard it
    touches is one it creates itself starting at ``start_shard_idx``, refusing if that
    name is already taken. See the module docstring for why that matters for a
    hardlinked clone.

    ``dry_run=True`` runs the exact same rolling/alignment bookkeeping (so the reported
    plan is exact, not estimated) but skips ``open()``/file writes and does not require
    write permission on ``out_dir``.
    """

    def __init__(self, out_dir: str, *, shard_limit: int, start_global: int,
                 start_shard_idx: int, dry_run: bool = False):
        assert shard_limit % ALIGN == 0, "shard_limit must be a multiple of ALIGN"
        assert start_global % ALIGN == 0, "graft must start on an aligned offset"
        self.out_dir = out_dir
        self.shard_limit = shard_limit
        self.dry_run = dry_run
        self._tensors: list[dict] = []
        self._new_shards: list[dict] = []
        self._global = start_global
        self._shard_idx = start_shard_idx - 1  # pre-incremented in _roll
        self._f = None
        self._started = False
        self._shard_start = start_global
        self._cur = 0

    def _roll(self) -> None:
        if self._started:
            self._new_shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                     "global_off": self._shard_start, "nbytes": self._cur})
            if self._f is not None:
                self._f.close()
                self._f = None
        self._shard_idx += 1
        path = os.path.join(self.out_dir, _SHARD_FMT.format(self._shard_idx))
        if os.path.exists(path):
            raise GraftError(
                f"refusing to graft: target shard file already exists: {path}"
            )
        self._shard_start = self._global
        self._cur = 0
        self._started = True
        if not self.dry_run:
            self._f = open(path, "wb")

    def _write_raw(self, data) -> None:
        if not self._started:
            self._roll()
        off, n = 0, len(data)
        while off < n:
            if self._cur == self.shard_limit:
                self._roll()
            take = min(n - off, self.shard_limit - self._cur)
            if self._f is not None:
                self._f.write(data[off:off + take])
            off += take
            self._cur += take
            self._global += take

    def add_tensor(self, name: str, tensor: torch.Tensor, kind: str = "weight") -> str:
        """Same contract as ``FTWWriter.add_tensor``. Returns the sha256 hex digest of
        the tensor's raw bytes as written, for the caller's post-write self-verify."""
        t = tensor.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8)
        nbytes = int(raw.numel())
        if not self._started or (nbytes <= self.shard_limit
                                 and self._cur + nbytes > self.shard_limit):
            self._roll()
        global_off = self._global
        assert global_off % ALIGN == 0, "tensor start must be aligned (invariant)"
        raw_bytes = raw.numpy().tobytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        self._write_raw(memoryview(raw_bytes))
        self._tensors.append({"name": name, "kind": kind, "dtype": _dtype_str(t.dtype),
                              "shape": list(t.shape), "global_off": global_off, "nbytes": nbytes})
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))
        return digest

    def finalize(self) -> tuple[list[dict], list[dict], int]:
        """Returns ``(new_tensor_entries, new_shard_entries, new_total_bytes)``."""
        if self._started:
            self._new_shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                     "global_off": self._shard_start, "nbytes": self._cur})
            if self._f is not None:
                self._f.close()
                self._f = None
        return self._tensors, self._new_shards, self._global


# ============================== clone ==============================


def clone_ftw_dir(src_dir: str, dst_dir: str) -> None:
    """Hardlink-clone an FTW checkpoint directory (the ``cp -al`` effect): every file
    under ``src_dir`` gets a new directory entry under ``dst_dir`` pointing at the SAME
    inode, so the clone costs ~0 extra disk bytes. See the module docstring for why
    grafting into the clone afterwards never writes through one of those shared inodes.
    """
    if os.path.exists(dst_dir):
        raise GraftError(f"--clone-to target already exists: {dst_dir}")
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(out_root, exist_ok=True)
        for name in files:
            os.link(os.path.join(root, name), os.path.join(out_root, name))


# ============================== self-verify ==============================


def _read_tensor_bytes(reader: FTWReader, entry: dict) -> bytes:
    buf = bytearray(_align_up(entry["nbytes"]))
    reader.read_into(memoryview(buf), entry)
    return bytes(buf[: entry["nbytes"]])


def _sample_entries(tensors: list[dict], k: int) -> list[dict]:
    if len(tensors) <= k:
        return list(tensors)
    half = max(1, k // 2)
    seen: set[str] = set()
    out = []
    for t in tensors[:half] + tensors[-half:]:
        if t["name"] not in seen:
            seen.add(t["name"])
            out.append(t)
    return out


def _hash_existing_sample(ftw_dir: str, tensors: list[dict], k: int) -> dict[str, str]:
    sample = _sample_entries(tensors, k)
    if not sample:
        return {}
    reader = FTWReader(ftw_dir)
    try:
        return {e["name"]: hashlib.sha256(_read_tensor_bytes(reader, e)).hexdigest() for e in sample}
    finally:
        reader.close()


def _self_verify(ftw_dir: str, new_digests: dict[str, str], pre_hashes: dict[str, str]) -> dict:
    """Reopen the just-written checkpoint fresh and prove: (1) every new tensor reads
    back byte-identical to what was written, and it is reachable as ``kind="weight"``
    (cheap in-memory check -- no data read for this part, so this stays fast even
    against a checkpoint with thousands of unrelated entries); (2) the pre-graft sample
    of pre-existing tensors is unchanged; (3) the shard list is still gap-free."""
    reader = FTWReader(ftw_dir)
    try:
        weight_names = {e["name"] for e in reader.entries("weight")}
        missing = sorted(set(new_digests) - weight_names)
        if missing:
            raise GraftError(
                f"self-verify FAILED: new tensor(s) not reachable as kind='weight': {missing[:5]}"
            )

        for name, expected in new_digests.items():
            entry = reader.tensors.get(name)
            if entry is None:
                raise GraftError(f"self-verify FAILED: new tensor {name!r} missing from reopened index")
            got = hashlib.sha256(_read_tensor_bytes(reader, entry)).hexdigest()
            if got != expected:
                raise GraftError(f"self-verify FAILED: new tensor {name!r} hash mismatch after re-read")

        for name, expected in pre_hashes.items():
            entry = reader.tensors[name]
            got = hashlib.sha256(_read_tensor_bytes(reader, entry)).hexdigest()
            if got != expected:
                raise GraftError(
                    f"self-verify FAILED: pre-existing tensor {name!r} changed after graft"
                )

        shards = sorted(reader.index["shards"], key=lambda s: s["global_off"])
        pos = 0
        for sh in shards:
            if sh["global_off"] != pos:
                raise GraftError(f"self-verify FAILED: post-graft shard list has a gap at {sh['file']!r}")
            pos += sh["nbytes"]
        if pos != reader.index["total_bytes"]:
            raise GraftError("self-verify FAILED: post-graft shards do not sum to total_bytes")
    finally:
        reader.close()
    return {"new_ok": len(new_digests), "existing_ok": len(pre_hashes)}


def _tool_commit() -> str | None:
    """Best-effort git commit of this tool's own source tree, for index provenance."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        logger.warning("could not resolve graft tool's git commit", exc_info=True)
    return None


# ============================== orchestration ==============================


def graft_vision(ftw_dir: str, source_dir: str, *, dtype: torch.dtype = torch.bfloat16,
                  dry_run: bool = False, verify_sample: int = _DEFAULT_VERIFY_SAMPLE) -> dict:
    """Append ``source_dir``'s vision tower to the FTW checkpoint at ``ftw_dir``, in
    place: a new shard is written and the index is extended (temp file + ``os.replace``,
    never edited in place). ``dry_run=True`` validates preconditions and computes the
    exact plan (new shard filename(s), byte counts, resulting ``total_bytes``) without
    writing anything. Raises :class:`GraftError` on any precondition or self-verify
    failure. Returns a plan/result dict (see the ``plan`` keys below; ``dry_run`` runs
    omit ``verified_new``/``verified_sample_existing``).
    """
    index = _load_index(ftw_dir)
    shard_limit = _validate_index(index, ftw_dir)
    total_bytes = index["total_bytes"]
    next_idx = _next_shard_idx(index["shards"])
    _check_target_shard_absent(ftw_dir, next_idx)

    pre_hashes: dict[str, str] = {}
    pre_shard_stat: dict[str, tuple] = {}
    if not dry_run:
        pre_hashes = _hash_existing_sample(ftw_dir, index["tensors"], verify_sample)
        pre_shard_stat = _stat_shards(ftw_dir, index["shards"])

    writer = _GraftWriter(ftw_dir, shard_limit=shard_limit, start_global=total_bytes,
                          start_shard_idx=next_idx, dry_run=dry_run)

    digests: dict[str, str] = {}
    for name, tensor in _iter_vision_from_disk(source_dir, dtype=dtype):
        digests[name] = writer.add_tensor(name, tensor, kind="weight")
    n_vision = len(digests)
    if n_vision == 0:
        raise GraftError(
            f"{source_dir}: no model.visual.* tensors found -- nothing to graft "
            "(this source checkpoint may not ship a vision tower)"
        )

    new_tensors, new_shards, new_total_bytes = writer.finalize()

    plan = {
        "ftw_dir": ftw_dir,
        "source_dir": os.path.abspath(source_dir),
        "n_vision_tensors": n_vision,
        "bytes_written": new_total_bytes - total_bytes,
        "old_total_bytes": total_bytes,
        "new_total_bytes": new_total_bytes,
        "new_shards": new_shards,
    }
    if dry_run:
        return plan

    # Existing shard files must be byte-for-byte untouched by everything above -- this
    # is a correctness invariant of _GraftWriter (it never opens them), checked here
    # too as a hard belt-and-suspenders assertion before the index is ever rewritten.
    post_shard_stat = _stat_shards(ftw_dir, index["shards"])
    if post_shard_stat != pre_shard_stat:
        raise GraftError(
            "internal error: an existing shard file's (inode, size, mtime) changed "
            "during grafting -- refusing to rewrite the index"
        )

    new_index = dict(index)
    new_index["total_bytes"] = new_total_bytes
    new_index["tensors"] = index["tensors"] + new_tensors
    new_index["shards"] = index["shards"] + new_shards
    counts = dict(index.get("counts", {}))
    counts["weight"] = counts.get("weight", 0) + n_vision
    new_index["counts"] = counts
    new_index["vision_num_tensors"] = n_vision
    new_index["vision_grafted_from"] = os.path.abspath(source_dir)
    new_index["vision_graft_tool"] = "freetoken.checkpoint.graft_vision"
    new_index["vision_graft_commit"] = _tool_commit()
    new_index["vision_graft_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    _write_index(ftw_dir, new_index)

    verified = _self_verify(ftw_dir, digests, pre_hashes)
    plan["verified_new"] = verified["new_ok"]
    plan["verified_sample_existing"] = verified["existing_ok"]
    return plan


# ============================== CLI ==============================


def _format_plan(plan: dict) -> str:
    lines = [
        f"graft plan for {plan['ftw_dir']}",
        f"  source: {plan['source_dir']}",
        f"  vision tensors: {plan['n_vision_tensors']}",
        f"  bytes to write: {plan['bytes_written']} ({plan['bytes_written'] / (1 << 20):.2f} MiB)",
        f"  total_bytes: {plan['old_total_bytes']} -> {plan['new_total_bytes']}",
        "  new shard(s):",
    ]
    for sh in plan["new_shards"]:
        lines.append(f"    {sh['file']}  global_off={sh['global_off']}  nbytes={sh['nbytes']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, prog: str = "freetoken.checkpoint.graft_vision") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ftw", required=True, help="FTW checkpoint dir to extend")
    p.add_argument("--source", required=True,
                   help="HF checkpoint dir carrying the model.visual.* weights")
    p.add_argument("--clone-to", default=None,
                   help="hardlink-clone --ftw into this new dir first (os.link per file, "
                        "~0 extra bytes), then graft into the clone; --ftw itself is left "
                        "untouched. Ignored (no clone created) under --dry-run.")
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit; writes nothing, creates no clone")
    ns = p.parse_args(argv)

    if ns.dry_run:
        note = f" (as it would look after hardlink-cloning to {ns.clone_to!r})" if ns.clone_to else ""
        print(f"[dry-run] plan against {ns.ftw!r}{note}:")
        plan = graft_vision(ns.ftw, ns.source, dtype=_DTYPES[ns.dtype], dry_run=True)
        print(_format_plan(plan))
        print("  (dry run: nothing written, no clone created)")
        return 0

    target = ns.ftw
    if ns.clone_to:
        clone_ftw_dir(ns.ftw, ns.clone_to)
        print(f"cloned (hardlink) {ns.ftw!r} -> {ns.clone_to!r}")
        target = ns.clone_to

    t0 = time.perf_counter()
    plan = graft_vision(target, ns.source, dtype=_DTYPES[ns.dtype], dry_run=False)
    dt = time.perf_counter() - t0
    print(_format_plan(plan))
    print(f"  verified {plan['verified_new']} new + {plan['verified_sample_existing']} "
          f"sampled pre-existing tensor(s) byte-identical")
    print(f"  done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GraftError", "clone_ftw_dir", "graft_vision", "main",
]
