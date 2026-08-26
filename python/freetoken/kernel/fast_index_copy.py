from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


DEFAULT_NUM_BLOCKS = 4
SKIP_FAST_INDEX_COPY_ENV = "FREETOKEN_SKIP_FAST_INDEX_COPY"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _skip_fast_index_copy_enabled() -> bool:
    return os.getenv(SKIP_FAST_INDEX_COPY_ENV, "").strip().lower() in _TRUE_VALUES


# Kill switch for the pitched (payload-only) multi-bank copy. Set it and every
# caller reverts to transferring the full row pitch -- the behaviour that shipped
# before per-row copy widths existed.
UNPITCHED_COPY_ENV = "FREETOKEN_UNPITCHED_COPY"


def _unpitched_copy_enabled() -> bool:
    return os.getenv(UNPITCHED_COPY_ENV, "").strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=None)
def _jit_update_flag_module() -> Module:
    return load_jit(
        "fast_index_copy_update_flag",
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("update_copy_flag", "update_copy_flag")],
    )


@lru_cache(maxsize=None)
def _jit_fast_index_copy_module(
    *,
    feature_size: int,
    worker_threads: int,
    worker_feature_size: int,
    num_block: int,
) -> Module:
    args = make_cpp_args(
        feature_size,
        worker_threads,
        worker_feature_size,
        1024,
        num_block,
        1,
    )
    return load_jit(
        "fast_index_copy",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[
            ("launch", f"&FastIndexCopyKernel<{args}>::run"),
            ("launch_high", f"&FastIndexCopyKernel<{args}>::run_high"),
            ("launch_normal", f"&FastIndexCopyKernel<{args}>::run_normal"),
        ],
    )


def _default_worker_threads(feature_size: int) -> int:
    if feature_size <= 1024:
        return 8
    if feature_size <= 2048:
        return 16
    return 32


def _shrink_worker_feature_size(feature_size: int, worker_feature_size: int) -> int:
    if feature_size < worker_feature_size:
        worker_feature_size = feature_size
    while feature_size % worker_feature_size != 0 and worker_feature_size > 128:
        worker_feature_size //= 2
    return worker_feature_size


def default_worker_args(feature_size: int) -> tuple[int, int, int]:
    """(worker_threads, worker_feature_size, num_block) of the default call path.

    The AOT spec builder (kernel/aot.py) compiles with exactly these values, so a
    default ``fast_index_copy_jit`` call hits the prebuilt cache by name.
    """
    return (
        _default_worker_threads(feature_size),
        _shrink_worker_feature_size(feature_size, 2048),
        DEFAULT_NUM_BLOCKS,
    )


def fast_index_copy_jit(
    dst: torch.Tensor,
    dst_indices: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    worker_threads: int | None = None,
    worker_feature_size: int = 2048,
    num_block: int | None = None,
    priority: str | None = None,
    sync_flag: torch.Tensor | None = None,
) -> None:
    num_dst_feature = math.prod(dst.shape[1:])
    num_src_feature = math.prod(src.shape[1:])
    assert num_src_feature == num_dst_feature

    # Debug/perf ablation: keep miss bookkeeping intact, but make the copy free to
    # approximate a zero-copy-miss runtime. Outputs are only meaningful if callers
    # already have valid cache contents for the requested indices.
    if _skip_fast_index_copy_enabled():
        return

    dst = dst.as_strided(size=(dst.size(0), num_dst_feature), stride=(num_dst_feature, 1))
    src = src.as_strided(size=(src.size(0), num_src_feature), stride=(num_src_feature, 1))

    feature_size = dst.size(-1) * dst.element_size()
    num_block = num_block or DEFAULT_NUM_BLOCKS
    worker_threads = worker_threads or _default_worker_threads(feature_size)
    worker_feature_size = _shrink_worker_feature_size(feature_size, worker_feature_size)
    assert worker_threads in (8, 16, 32)
    assert feature_size % worker_feature_size == 0

    module = _jit_fast_index_copy_module(
        feature_size=feature_size,
        worker_threads=worker_threads,
        worker_feature_size=worker_feature_size,
        num_block=num_block,
    )
    if priority is None:
        module.launch(dst, dst_indices, src, src_indices, num_indices)
        return

    assert priority in ("high", "normal")
    assert sync_flag is not None
    if priority == "high":
        module.launch_high(dst, dst_indices, src, src_indices, num_indices, sync_flag)
        return
    module.launch_normal(dst, dst_indices, src, src_indices, num_indices, sync_flag)


@lru_cache(maxsize=None)
def _jit_fast_index_copy_multi_module(*, num_threads: int, blocks_per_bank: int) -> Module:
    args = make_cpp_args(num_threads, blocks_per_bank)
    return load_jit(
        "fast_index_copy_multi",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("launch", f"&MultiIndexCopyKernel<{args}>::run")],
    )


def fast_index_copy_multi_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    copy_bytes: torch.Tensor | None = None,
    row_pitch: torch.Tensor | None = None,
    num_threads: int = 1024,
    blocks_per_bank: int = 8,
) -> None:
    """Fused multi-bank index copy: copy the same rows for every bank in ONE launch.

    The copy is host->device PCIe-bandwidth bound (~31 GB/s measured, vs ~3 TB/s HBM), so what
    saturates the link is the TOTAL in-flight request count, blocks_per_bank*num_threads
    (~4096 threads/bank is the knee; blocks and threads are interchangeable -- 8x1024 ==
    16x512 == 4x1024 all measure the same across small/medium/large-expert workloads). PCIe
    is the bottleneck, not the GPU, so few blocks already feed it; the split only matters at
    the extreme (~100k+ threads bumps the zero-copy launch overhead). 8x1024 = 8192/bank sits
    ~2x past the knee. Launch count is one regardless of grid width, so the zero-copy launch-
    overhead win (and a strict >= over the legacy per-bank kernel across all workloads) holds.

    ``dst_ptrs``/``src_ptrs``/``feat_bytes`` are int64 [num_banks] device tensors built
    once by the caller (the per-bank slot-cache base addr, host-source base addr, and
    per-row byte size). Every bank's per-row byte size must be a multiple of 16, and the
    base addresses 16-byte aligned (true for contiguous torch allocations of these banks).

    ``copy_bytes`` + ``row_pitch`` (optional, given together, same [num_banks] int64
    device shape) move only the PAYLOAD of each weight row. One bank row is a whole
    expert: a stack of ``feat_bytes // row_pitch`` weight rows at a uniform byte pitch.
    A mixed-quant bank sets that pitch from the widest ggml type it holds, and the GEMV
    never reads past a row's own native packed width, so the tail need not cross PCIe --
    it is simply left as it was in the destination slot, which is exactly as unread.
    ``copy_bytes`` is the per-weight-row payload (a multiple of 16, at most
    ``row_pitch``); ``row_pitch`` must divide ``feat_bytes``. Omit both (or set
    ``FREETOKEN_UNPITCHED_COPY=1``) for the historical full-row copy, which is a
    separate kernel specialization rather than a runtime branch.
    """
    if _skip_fast_index_copy_enabled():
        return
    if copy_bytes is not None and _unpitched_copy_enabled():
        # Escape hatch for the most load-bearing kernel in the decode loop: one env
        # var puts every caller back on the byte-for-byte pre-existing path.
        copy_bytes = row_pitch = None
    assert (copy_bytes is None) == (row_pitch is None), (
        "fast_index_copy_multi: copy_bytes and row_pitch must be given together"
    )
    module = _jit_fast_index_copy_multi_module(
        num_threads=num_threads, blocks_per_bank=blocks_per_bank
    )
    module.launch(
        dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices, num_indices,
        copy_bytes, row_pitch,
    )


def update_copy_flag_jit(sync_flag: torch.Tensor, delta: int) -> None:
    assert sync_flag.is_cuda
    assert sync_flag.numel() == 1
    assert sync_flag.dtype == torch.int32
    _jit_update_flag_module().update_copy_flag(sync_flag, delta)
