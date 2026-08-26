"""Convert an HF safetensors checkpoint into a FreeToken Weight (FTW) checkpoint.

Model-agnostic: it drives the *existing* per-model loaders once and stores their output, so
no per-model conversion code is needed.

* dense weights = exactly what ``load_weight(include_moe_experts=...)`` yields (post
  fusion/TP-shard) -> ``kind="weight"``; at load they feed ``model.load_state_dict``.
  ``mtp.*`` / ``model.visual.*`` are dropped even if a loader leaks them.
* offload experts = exactly what ``load_expert_banks(..., layer_sink=)`` produces (post
  backend-repack pinned banks + alpha scale vectors) -> ``kind="experts_bank"`` (alphas are
  told apart at load by their reserved names, so they need no separate kind).
* n-gram tables (Qwen) = original HF tensors -> ``kind="ngram"`` (not concat; banked at
  serve time by ``moe/ngram_bank.py`` via ``iter_ftw_ngrams``).

The output directory is a self-contained checkpoint (config + tokenizer copied), so you can
point ``--model`` straight at it; the load path auto-detects the FTW and reads it (FTW).
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import threading

import torch

from .ftw import DEFAULT_SHARD_LIMIT, FTWWriter, layer_bank_entry_name
from .qwen_layout import (
    DEST_EXPERT,
    DEST_NGRAM,
    DEST_SKIP,
    classify_tensor,
    disk_free_bytes,
    extra_safetensor_files,
    is_mtp_expert_tensor,
    is_mtp_tensor,
    is_qwen4_exp_config,
    is_wrapper_config,
    iter_shard_tensor_metas,
    load_weight_map,
    looks_like_ngram_shard,
    read_safetensors_header,
    tensor_nbytes_from_meta,
)

# Conversion guard: refuse to start if less than this much free space would remain on the
# output volume after the estimated write. A conversion that runs out of disk mid-write
# leaves a half-written checkpoint dir that LOOKS like a valid FTW (index.json present)
# but is silently truncated -- catching this before any bytes are written is much cheaper
# than discovering it after a 100+ GiB write.
_MIN_FREE_BYTES_AFTER = 20 << 30  # ~20 GiB

# Machine-readable convert progress for a supervising process (e.g. a GUI frontend parses
# these `FTCONVERT <phase> <done> <total>` stdout lines to drive its convert bar). Gated by
# FREETOKEN_CONVERT_PROGRESS=1 so plain CLI use isn't spammed; the human tqdm bars stay on
# stderr. Phases: `dense` (indeterminate, done=total=0), `experts` (byte totals), `finalize`.
_EMIT_PROGRESS = os.environ.get("FREETOKEN_CONVERT_PROGRESS") == "1"


def _progress(phase: str, done: int = 0, total: int = 0) -> None:
    if _EMIT_PROGRESS:
        print(f"FTCONVERT {phase} {done} {total}", flush=True)


def _source_fingerprint(model_path: str, model_config, *, device) -> str:
    """Identity of (checkpoint + quant + GPU capability), stored in the FTW index so
    it's clear what an FTW was built from. Cheap (stat only)."""
    h = hashlib.sha256()
    h.update(f"quant={getattr(model_config, 'expert_quant', None)}|".encode())
    h.update(f"arch={getattr(model_config, 'architectures', None)}|".encode())
    try:  # nvfp4 marlin/b12x layout depends on compute capability
        h.update(f"cc={torch.cuda.get_device_capability(device)}|".encode())
    except Exception:
        pass
    files = sorted(
        glob.glob(os.path.join(model_path, "*.safetensors")) + glob.glob(os.path.join(model_path, "*.gguf"))
    )
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    # q2_k_ud/q4_k_ud: the routed experts are read from a SEPARATE GGUF (--expert-gguf),
    # not model_path, so the loop above never sees it -- fold in its shard set (resolved
    # to real files, so a differently-named-but-identical split set still matches).
    # q2_k_ud additionally folds in the MXFP4->Q2_K re-encoder's version, since that's the
    # one place THAT provider changes the numbers rather than copying bytes: a re-encoder
    # change must invalidate an old FTW even though the source GGUF itself didn't change.
    # q4_k_ud never re-encodes (every ggml type it uses is natively kernel-supported), so
    # it has no analogous version to fold in.
    gguf_path = None
    for attr in ("dsv4_args", "qwen4_args"):
        gguf_path = getattr(getattr(model_config, attr, None), "expert_gguf_path", None)
        if gguf_path:
            break
    if gguf_path:
        from freetoken.models.gguf.reader import _split_shard_paths

        if getattr(model_config, "dsv4_args", None) is not None:
            from freetoken.models.deepseek_v4.gguf_experts import Q2K_REENCODE_VERSION

            h.update(f"q2k_reencode_version={Q2K_REENCODE_VERSION}|".encode())
        for f in sorted(_split_shard_paths(gguf_path) or [gguf_path]):
            st = os.stat(f)
            h.update(f"expert_gguf:{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    return h.hexdigest()[:16]


def _estimate_output_bytes(model_path: str, expert_gguf: str | None) -> int:
    """Cheap (header-only + file-stat-only, no tensor data read) upper-bound estimate of
    the FTW's on-disk payload, for the pre-flight disk-space guard.

    Sums safetensors header ``nbytes`` for everything EXCEPT the target's own routed
    experts (``DEST_EXPERT``) when ``--expert-gguf`` replaces them with a (much smaller)
    quantized bank -- in that case the GGUF shard set's real file size is added instead
    (a stat() per shard, not a parse). mtp.* tensors are always counted in full: this
    converter preserves them losslessly regardless of --expert-gguf (see
    ``_iter_mtp_from_disk``). Under-counts slightly (ignores the FTW's own per-tensor
    4096-byte alignment padding and the GGUF's block-quant metadata overhead) but is
    within a few percent of the real total -- fine for a coarse pre-flight guard, not
    used for anything else.
    """
    weight_map = load_weight_map(model_path)
    shards = sorted(set(weight_map.values()))
    total = 0
    for name, _shard, _meta, nbytes in iter_shard_tensor_metas(model_path, shards):
        if is_mtp_tensor(name):
            total += nbytes
            continue
        dest = classify_tensor(name)
        if dest == DEST_SKIP:
            continue
        if dest == DEST_EXPERT and expert_gguf:
            continue  # replaced below by the GGUF's own (smaller) shard bytes
        total += nbytes
    if expert_gguf:
        from freetoken.models.gguf.reader import _split_shard_paths

        for f in sorted(_split_shard_paths(expert_gguf) or [expert_gguf]):
            total += os.path.getsize(f)
    return total


def _guard_output_dir(out_dir: str, model_path: str, expert_gguf: str | None) -> None:
    """Refuse to start a conversion that would overwrite an existing dir or run the
    output volume below ``_MIN_FREE_BYTES_AFTER``. Both checks are cheap (no tensor data
    read) and run before any bytes are written."""
    if os.path.exists(out_dir):
        raise SystemExit(
            f"{out_dir} already exists -- refusing to write over it (a half-written FTW "
            f"dir looks valid but is silently truncated; remove it first if you intend "
            f"to rebuild)"
        )
    estimated = _estimate_output_bytes(model_path, expert_gguf)
    probe_dir = os.path.dirname(os.path.abspath(out_dir)) or os.sep
    os.makedirs(probe_dir, exist_ok=True)
    free = disk_free_bytes(probe_dir)
    remaining = free - estimated
    if remaining < _MIN_FREE_BYTES_AFTER:
        raise SystemExit(
            f"refusing to start: estimated FTW size {estimated / (1 << 30):.1f} GiB, "
            f"only {free / (1 << 30):.1f} GiB free on the volume holding {out_dir!r} -- "
            f"would leave {remaining / (1 << 30):.1f} GiB free, below the "
            f"{_MIN_FREE_BYTES_AFTER / (1 << 30):.0f} GiB safety margin"
        )
    logger_msg = (
        f"disk guard: estimated FTW size {estimated / (1 << 30):.1f} GiB, "
        f"{free / (1 << 30):.1f} GiB free -> {remaining / (1 << 30):.1f} GiB free "
        f"after (>= {_MIN_FREE_BYTES_AFTER / (1 << 30):.0f} GiB margin), proceeding"
    )
    print(logger_msg, flush=True)


def _hf_aliases(name: str) -> set[str]:
    """Raw HF key and the language_model-stripped form ``iter_weights`` may yield."""
    aliases = {name}
    if name.startswith("model.language_model."):
        aliases.add("model." + name[len("model.language_model."):])
    elif name.startswith("language_model."):
        aliases.add("model." + name[len("language_model."):])
    elif name.startswith("model.") and not name.startswith("model.language_model."):
        aliases.add("model.language_model." + name[len("model."):])
    return aliases


def _iter_ngram_from_disk(model_path: str, *, skip_names: set[str]):
    """Stream n-gram tables that ``iter_weights`` is contracted to skip.

    Preserve original HF keys and tensor bytes (no concat): ``moe/ngram_bank.py``
    owns bank-ification / mmap. Concat would invent an FTW-only layout that
    package cannot see in the source snapshot.

    Ordering: this generator is consumed on the convert *main thread* before
    ``load_expert_banks`` starts its reader threads. ``FTWWriter`` is not
    thread-safe; writing n-gram here (not from a sink callback) keeps the
    writer exclusive until the expert ``_ConvertSink`` lock takes over.
    """
    import safetensors

    weight_map = load_weight_map(model_path)
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if skip_names.intersection(_hf_aliases(name)):
            continue
        if classify_tensor(name) == DEST_NGRAM:
            by_shard.setdefault(shard, []).append(name)
    for fname in extra_safetensor_files(model_path, weight_map):
        path = os.path.join(model_path, fname)
        try:
            hdr = read_safetensors_header(path)
        except (OSError, ValueError):
            continue
        force = looks_like_ngram_shard(fname)
        for name, meta in hdr.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            if skip_names.intersection(_hf_aliases(name)):
                continue
            if force or classify_tensor(name) == DEST_NGRAM:
                by_shard.setdefault(fname, []).append(name)
    for shard, names in sorted(by_shard.items()):
        path = os.path.join(model_path, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in names:
                yield name, f.get_tensor(name)


def _iter_mtp_from_disk(model_path: str, *, skip_names: set[str]):
    """Stream ``mtp.*`` (MTP drafter head) tensors, preserved for a later phase.

    ``load_weight``/``iter_weights`` never yield ``mtp.*`` -- ``qwen4_exp/weight.py``'s
    ``classify_key`` routes them to the ``mtp_dense`` / ``mtp_expert_bank`` categories,
    both of which ``iter_weights`` skips (no ``Qwen4ExpForCausalLM.mtp`` submodule exists
    yet to receive them via ``load_state_dict`` -- wiring up the drafter's forward pass is
    a later phase; see ``python/freetoken/models/qwen4_exp/mtp.py``). Preserve the
    original checkpoint keys and tensor bytes verbatim (no concat/rename -- same
    no-invented-layout contract as ``_iter_ngram_from_disk``) under two new FTW kinds so
    the ~2.5 GiB drafter head survives conversion losslessly without being wired into the
    runtime model:
      * kind="mtp_dense"  -- 29 bf16 tensors (fc_embedding/fc_hidden + norms, the mixer,
        and one MTP decoder layer: self_attn/indexer, MoE gate + shared_expert(_gate),
        and two hyper_connections).
      * kind="mtp_expert" -- 3072 fp8 tensors, the drafter's own 512-expert routed bank
        (mtp.layers.0.mlp.experts.*), same block-fp8 layout as the target's routed
        experts but a separate bank -- NOT run through ``load_expert_banks``/GGUF (that
        pipeline sources the *target's* experts only; --expert-gguf carries no mtp.*
        rows).
    Neither kind is read by ``iter_ftw_weights``'s default ``kinds=("weight",)``, so this
    is inert until a future phase reads them back by kind (same pattern ``kind="ngram"``
    already uses for the n-gram table).

    Ordering: like ``_iter_ngram_from_disk``, this runs on the convert *main thread*
    before ``load_expert_banks`` starts its reader threads -- ``FTWWriter`` is not
    thread-safe until the expert ``_ConvertSink`` lock takes over.
    """
    import safetensors

    weight_map = load_weight_map(model_path)
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for name, shard in weight_map.items():
        if name in skip_names or not is_mtp_tensor(name):
            continue
        kind = "mtp_expert" if is_mtp_expert_tensor(name) else "mtp_dense"
        by_shard.setdefault(shard, []).append((name, kind))
    for shard, items in sorted(by_shard.items()):
        path = os.path.join(model_path, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name, kind in items:
                yield name, kind, f.get_tensor(name)


def _require_qwen4_exp_registered(hf_cfg) -> None:
    """Fail loud if this is qwen4_exp but ``models/qwen4_exp`` is not on the registry.

    The converter stays model-agnostic and will not invent a loader; the other
    worker owns ``parse_config`` / ``iter_weights`` / ``setup_offload_expert_banks``.
    """
    from freetoken.models.register import get_model_spec

    archs = getattr(hf_cfg, "architectures", None) or []
    if not archs:
        raise SystemExit(f"{getattr(hf_cfg, 'model_type', '?')}: config has no architectures")
    try:
        get_model_spec(archs[0])
    except ValueError:
        cfg_dict = {
            "architectures": list(archs),
            "model_type": getattr(hf_cfg, "model_type", None),
            "text_config": getattr(hf_cfg, "text_config", None),
        }
        if is_qwen4_exp_config(cfg_dict) or is_wrapper_config(cfg_dict):
            raise SystemExit(
                f"architecture {archs[0]!r} is not registered. qwen4_exp must export "
                f"parse_config/iter_weights/setup_offload_expert_banks and be listed in "
                f"models/register.py -- see /root/ftw_qwen_plan.md"
            ) from None
        raise

# Checkpoint metadata to carry over so the FTW dir is a usable checkpoint on its own.
# (Weight shards + the safetensors index are intentionally NOT copied.)
# Everything that is NOT a weight shard is metadata we carry over verbatim. A whitelist
# misses model-specific layouts (e.g. DSV4's inference/config.json + encoding/ live in
# subdirs), so we copy every non-weight file preserving its relative path instead.
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ftw")  # .ftw: a nested FTW, not source
_SKIP_NAMES = ("model.safetensors.index.json",)  # indexes shards the FTW replaces
# .freetoken_expert_cache: the legacy per-bank cache (can be tens of GB of stale .bin)
_SKIP_DIRS = (".git", ".cache", ".freetoken_expert_cache")


def _copy_metadata(model_path: str, out_dir: str) -> list[str]:
    """Copy all non-weight files (config, tokenizer, remote-code, nested model configs)
    preserving directory structure, so the FTW dir is a self-contained checkpoint."""
    if os.path.isfile(model_path):
        # A single-file source has no sibling metadata to walk: a .gguf carries its config
        # AND tokenizer in its own KV section. Emit a metadata-only copy (header + KV, no
        # weight data) the FTW dir resolves those from. We deliberately do NOT sweep the
        # file's parent dir -- an HF gguf snapshot dir can hold unrelated blobs.
        from freetoken.models.gguf.reader import (
            FTW_METADATA_GGUF,
            is_gguf_path,
            write_metadata_gguf,
        )

        if is_gguf_path(model_path):
            os.makedirs(out_dir, exist_ok=True)
            write_metadata_gguf(model_path, os.path.join(out_dir, FTW_METADATA_GGUF))
            return [FTW_METADATA_GGUF]
        return []

    out_abs = os.path.abspath(out_dir)
    copied = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and os.path.abspath(os.path.join(root, d)) != out_abs]
        for name in files:
            if name.endswith(_WEIGHT_SUFFIXES) or name in _SKIP_NAMES:
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, model_path)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    return copied


class _ConvertSink:
    """Layer-completion sink for ``load_expert_banks(layer_sink=...)``: writes each
    completed layer's banks as their own FTW entries immediately (name
    ``f"{bank_name}#L{layer_id:05d}"``, kind ``"experts_bank"``) and releases them, so
    conversion RAM peaks at ~in-flight layers instead of the whole bank set.

    Only engaged for streamable formats -- ``ExpertBanks.streamed`` reports whether this
    actually fired; if not, the caller falls back to the materialize-and-write path
    instead. The progress bar is created lazily on the first call, so a format that never
    streams never shows one.

    ``FTWWriter`` buffers file/shard state and is not thread-safe; completion callbacks
    can fire from the loader's own reader threads, so the write+release is serialized
    under one lock (disk-bound anyway).
    """

    def __init__(self, writer: FTWWriter, desc: str = "Converting expert banks") -> None:
        self._writer = writer
        self._desc = desc
        self._bar = None
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self.n_written = 0
        self.n_bytes = 0

    def __call__(self, layer_id: int, banks: dict) -> None:
        with self._lock:
            assert layer_id not in self._seen, f"layer {layer_id} streamed to the sink twice"
            self._seen.add(layer_id)
            if self._bar is None:
                from freetoken.utils.progress import byte_bar

                self._bar = byte_bar(0, self._desc)  # total unknown up front (streamed)
            nbytes = 0
            for bank_name, bank in banks.items():
                self._writer.add_tensor(
                    layer_bank_entry_name(bank_name, layer_id), bank.tensor, kind="experts_bank"
                )
                nbytes += bank.nbytes
                bank.release()
                self.n_written += 1
            self.n_bytes += nbytes
            self._bar.update(nbytes)
            # Cumulative BYTES (not the bank count): the supervisor maps this against the
            # known expert-pool size for a smooth phase-budgeted bar. Total stays 0 (unknown
            # up front while streaming); the materialize path below emits a real total.
            _progress("experts", self.n_bytes, 0)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @property
    def num_layers(self) -> int:
        return len(self._seen)


def convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    moe_backend: str = "offload",
    shard_limit: int = DEFAULT_SHARD_LIMIT,
    device: str | None = None,
    expert_gguf: str | None = None,
) -> dict:
    """Write ``model_path`` as an FTW checkpoint at ``out_dir``. Returns the index dict.

    The FTW format is TP-agnostic and conversion runs single-process, so the resulting
    checkpoint records no TP layout and loads independently of the runtime TP setting.

    ``expert_gguf``: DeepSeek-V4 or qwen4_exp only, mirrors ``--expert-gguf`` on the
    server -- convert the routed experts from this GGUF (DSV4: q2_k_ud, IQ2_XS/IQ3_XXS
    native rows, MXFP4 down rows re-encoded to Q2_K; qwen4_exp: q4_k_ud, Q4_K/Q5_K
    gate_up and Q5_1/Q8_0 down native rows, no re-encoding) instead of ``model_path``'s
    own expert weights. Dense weights (and everything else) still come from
    ``model_path``; the GGUF supplies only the offload expert banks. The converted FTW
    dir embeds the banks and their per-layer quant-type table, so a warm boot needs no
    re-parse of the (multi-GB, split-shard) GGUF -- but ``--expert-gguf`` must still be
    passed at boot (cheap: a path check, not a parse) so ``ModelConfig.expert_quant``
    resolves the same way it would on a cold boot; see ``engine.engine._apply_expert_gguf``.
    """
    from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.models.weight import load_weight
    from freetoken.moe.expert_banks import load_expert_banks
    from .ftw import is_ftw_checkpoint

    if is_ftw_checkpoint(model_path):
        raise SystemExit(f"{model_path} is already an FTW checkpoint")
    _guard_output_dir(out_dir, model_path, expert_gguf)
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
        tp = try_get_tp_info()
    elif tp.size != 1:
        raise SystemExit(
            f"FTW conversion runs single-process and the format records no TP layout, "
            f"but TP is already set to size={tp.size}"
        )

    # Recognize wrapper configs (text_config) before EngineConfig so a missing
    # qwen4_exp registry entry fails with the contract, not a raw KeyError.
    from freetoken.utils import cached_load_hf_config

    hf_cfg = cached_load_hf_config(model_path)
    _require_qwen4_exp_registered(hf_cfg)
    wrapper = getattr(hf_cfg, "text_config", None) is not None
    if wrapper:
        text = getattr(hf_cfg, "text_config", hf_cfg)
        print(
            f"FTW convert: wrapper config model_type={getattr(hf_cfg, 'model_type', None)!r} "
            f"text_config.model_type={getattr(text, 'model_type', None)!r} "
            f"arch={getattr(hf_cfg, 'architectures', None)}",
            flush=True,
        )

    # fp8_block (Qwen) conversion is CPU-capable: no NVFP4 dequant / marlin repack.
    # DSV4/NVFP4 still need a CUDA context. Do not force cuda:0 when CUDA is absent
    # -- that used to crash qwen4_exp convert on a CPU-only dry host.
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.zeros(1, device=dev)  # init CUDA context (nvfp4 backend pick / pinning)

    cfg = EngineConfig(model_path=model_path, tp_info=DistributedInfo(tp.rank, tp.size),
                       dtype=dtype, moe_backend=moe_backend, expert_gguf=expert_gguf)
    mc = cfg.model_config
    if expert_gguf:
        # DSV4 (q2_k_ud) or qwen4_exp (q4_k_ud) only. Passing a GGUF against any other
        # architecture has no args instance to stash the path on and no bank provider to
        # read it -- _apply_expert_gguf raises loudly in that case rather than guessing.
        # Must precede any expert_quant read below -- same ordering requirement as the
        # server's _adjust_config (see its comment): this rewrites model_config.expert_quant
        # and stashes the GGUF path on the format's args instance, in place, on the cached
        # ModelConfig instance cfg.model_config below will return.
        from freetoken.engine.engine import _apply_expert_gguf

        _apply_expert_gguf(cfg)
        mc = cfg.model_config
    # qwen4_exp model_type is "qwen4_exp" (no "moe" substring). is_moe is True only
    # when parse_config sets moe_enabled=True -- see /root/ftw_qwen_plan.md.
    offload = moe_backend == "offload" and getattr(mc, "is_moe", False)
    include_moe_experts = not offload

    from freetoken.utils.progress import byte_bar, count_bar

    writer = FTWWriter(out_dir, shard_limit=shard_limit)
    n_weight = n_bank = n_alpha = n_ngram = 0
    n_mtp_dense = n_mtp_expert = 0
    seen_ngram: set[str] = set()

    # 1) dense weights (host tensors; load straight to CPU to avoid GPU pressure)
    _progress("dense", 0, 0)  # phase start; per-tensor cumulative bytes follow (total unknown)
    dense_bytes = 0
    # adapt=False: store the model's STORAGE form, never the runtime form this converter
    # process's environment happens to resolve. An FTW has to be independent of the env it
    # was built under -- otherwise a serve-time switch (FREETOKEN_WO_A_FP8) gets baked in
    # at conversion and the checkpoint only serves the setting it was converted with. The
    # per-model adapt_weights hook re-applies at load (models/weight.py::load_weight).
    #
    # qwen4_exp / DSV4 pitfall: a model's adapt_weights (e.g. DSV4 wo_a FP8->bf16) must
    # NOT run here. qwen4_exp must not put env-dependent folds in iter_weights.
    for name, tensor in count_bar(load_weight(model_path, torch.device("cpu"),
                                              include_moe_experts=include_moe_experts,
                                              adapt=False),
                                  "Converting dense weights"):
        dest = classify_tensor(name)
        if dest == DEST_SKIP:
            # Defense in depth: even if iter_weights leaks mtp.* / model.visual.*, drop.
            continue
        if dest == DEST_EXPERT and offload:
            # Routed experts belong in layer_sink banks, not kind="weight".
            continue
        if dest == DEST_NGRAM:
            writer.add_tensor(name, tensor, kind="ngram")
            seen_ngram.add(name)
            n_ngram += 1
            dense_bytes += tensor.numel() * tensor.element_size()
            _progress("dense", dense_bytes, 0)
            continue
        writer.add_tensor(name, tensor, kind="weight")
        n_weight += 1
        dense_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes, 0)

    # 1b) n-gram pass-through: original HF keys, kind="ngram". iter_weights is
    # contracted to skip these (they are not state-dict params). Write them here
    # on the main thread before expert reader threads start (FTWWriter exclusivity).
    ngram_bytes = 0
    for name, tensor in count_bar(_iter_ngram_from_disk(model_path, skip_names=seen_ngram),
                                  "Converting n-gram tables"):
        writer.add_tensor(name, tensor, kind="ngram")
        seen_ngram.add(name)
        n_ngram += 1
        ngram_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes + ngram_bytes, 0)

    # 1c) MTP drafter pass-through: original HF keys, kind="mtp_dense" / "mtp_expert".
    # Preserved losslessly for a later phase (no Qwen4ExpForCausalLM.mtp submodule exists
    # yet to load_state_dict into, and the drafter's own expert bank is not part of
    # --expert-gguf's GGUF-sourced banks) -- see _iter_mtp_from_disk's docstring. Only
    # fires for a qwen4_exp source checkpoint that actually ships mtp.* keys; a no-op
    # (empty generator) otherwise, so this is a pure no-risk addition for every other
    # architecture (including DeepSeek-V4).
    mtp_bytes = 0
    for name, kind, tensor in count_bar(_iter_mtp_from_disk(model_path, skip_names=set()),
                                        "Converting MTP drafter head"):
        writer.add_tensor(name, tensor, kind=kind)
        if kind == "mtp_expert":
            n_mtp_expert += 1
        else:
            n_mtp_dense += 1
        mtp_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes + ngram_bytes + mtp_bytes, 0)

    # 2) offload expert banks (post-repack) + alpha scales (slow path auto-picks parallel/serial)
    quant_format = None
    num_layers = None
    quant_types = None  # q2_k_ud only: {"gate_up": [ggml type per layer], "down": [...]}
    if offload:
        # Streamable formats (bf16, ds_fp4, nvfp4 on the triton backend, gpt-oss mxfp4, q4_0,
        # qwen3_5 fp8/bf16-dequant) write each layer to its own FTW entry as it completes (via
        # the sink) instead of materializing the whole bank set first; the non-streamable ones
        # (nvfp4 marlin/b12x -- repack mutates the whole bank set in place after load) ignore
        # the sink. Which happened is per-provider (e.g. nvfp4's backend pick), so it's read
        # back from ExpertBanks.streamed, not guessed here.
        sink = _ConvertSink(writer)
        banks = load_expert_banks(model_path, mc, device=dev, dtype=dtype, layer_sink=sink)
        quant_format = banks.quant_format
        # q2_k_ud only: a layer's rows do not all share one ggml type, so the per-layer
        # type table travels as FTW metadata alongside the banks themselves (read back by
        # ftw.load_ftw_banks and re-attached to ExpertBanks.quant_types).
        quant_types = banks.quant_types
        if banks.streamed:
            sink.close()
            num_layers = sink.num_layers  # however many distinct layers the sink actually saw
            n_bank = sink.n_written
            assert num_layers > 0, (
                "provider reported streamed=True but the sink never fired -- the FTW "
                "would silently have no expert banks"
            )
            # Formats that fold their global scales (nvfp4 marlin/b12x) stream the weight
            # banks per layer but keep the alphas as flat [L*E] GPU vectors; write those as
            # flat reserved-name entries (same kind + names the materialize branch uses, so
            # the reader's reserved-name path reconstructs them identically).
            for an in ("gate_up_alpha", "down_alpha"):
                alpha = getattr(banks, an, None)
                if alpha is not None:
                    writer.add_tensor(an, alpha, kind="experts_bank")
                    n_alpha += 1
        else:
            # The on-disk format keeps one contiguous region per bank and the writer only
            # has whole-tensor add_tensor, so the per-layer sources reassemble into one
            # flat tensor (a per-bank host RAM spike during conversion).
            items = []
            for name, per_layer in banks.sources.items():
                if num_layers is None:
                    num_layers = len(per_layer)
                else:
                    assert len(per_layer) == num_layers, (name, len(per_layer), num_layers)
                items.append((name, torch.cat(per_layer, dim=0) if len(per_layer) > 1 else per_layer[0]))
            for an in ("gate_up_alpha", "down_alpha"):
                if getattr(banks, an, None) is not None:
                    items.append((an, getattr(banks, an)))
            total_bytes = sum(t.numel() * t.element_size() for _, t in items)
            bar = byte_bar(total_bytes, "Converting expert banks")
            done_bytes = 0
            _progress("experts", 0, total_bytes)
            for name, tensor in items:
                writer.add_tensor(name, tensor, kind="experts_bank")
                nbytes = tensor.numel() * tensor.element_size()
                bar.update(nbytes)
                done_bytes += nbytes
                _progress("experts", done_bytes, total_bytes)
                n_bank += name not in ("gate_up_alpha", "down_alpha")
                n_alpha += name in ("gate_up_alpha", "down_alpha")
            bar.close()

    _progress("finalize")  # writing shard index + copying config/tokenizer
    copied = _copy_metadata(model_path, out_dir)

    try:
        fingerprint = _source_fingerprint(model_path, mc, device=dev)
    except Exception:
        fingerprint = None

    index = writer.finalize({
        "source_model_path": os.path.abspath(model_path),
        "fingerprint": fingerprint,
        # quant_format records the actual on-disk bank layout (e.g. nvfp4_marlin vs
        # nvfp4_b12x): the suffix is a runtime backend pick (GPU capability / env), NOT in
        # config, and the stored bytes are physically repacked into it -- so it's kept and
        # read back at load (ftw.load_ftw_banks). dtype/moe_backend were dropped: each
        # tensor already carries its own dtype, and nothing reads a model-level backend.
        "quant_format": quant_format,
        # The reader takes num_layers from the model config (copied into this
        # checkpoint); recording it here too gives load_ftw_banks a cross-check that
        # the banks match the config they ship with. None for non-offload checkpoints.
        "expert_bank_num_layers": num_layers,
        # q2_k_ud only: {"gate_up": [ggml type per layer], "down": [...]} -- read back by
        # ftw.load_ftw_banks and re-attached to ExpertBanks.quant_types, since the GEMV
        # needs each layer's own decode type (see ExpertBanks.quant_types's docstring).
        # None for every format whose banks are one uniform type (the common case).
        "expert_bank_quant_types": quant_types,
        "counts": {
            "weight": n_weight,
            "experts_bank": n_bank + n_alpha,
            "ngram": n_ngram,
            "mtp_dense": n_mtp_dense,
            "mtp_expert": n_mtp_expert,
        },
        # n-gram tables are kind="ngram" (original HF names). ngram_bank.py reads
        # them via iter_ftw_ngrams / FTWReader.entries("ngram") -- not load_state_dict.
        "ngram_num_tensors": n_ngram,
        # MTP drafter head (preserved for a later phase, not yet wired into the runtime
        # model): kind="mtp_dense" (29 bf16, original HF names) + kind="mtp_expert" (3072
        # fp8, the drafter's own 512-expert routed bank). Neither is read by
        # iter_ftw_weights's default kinds=("weight",) -- see _iter_mtp_from_disk.
        "mtp_num_tensors": n_mtp_dense + n_mtp_expert,
        "copied_metadata": copied,
    })
    return index


__all__ = ["convert_checkpoint"]
