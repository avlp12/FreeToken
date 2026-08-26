from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _jit_memcpy2d_module() -> Module:
    return load_jit(
        "memcpy2d",
        cuda_files=["memcpy2d.cuh"],
        cuda_wrappers=[("memcpy2d", "&Memcpy2D::run")],
    )


def memcpy2d_h2d_jit(
    dst: torch.Tensor,
    src: torch.Tensor,
    width: int,
    *,
    stream: int | None = None,
) -> None:
    """Copy the leading ``width`` bytes of every row of ``src`` into ``dst``.

    Both tensors are contiguous and same-shaped; a "row" is the innermost dimension,
    so this transfers ``prod(shape[:-1])`` runs of ``width`` bytes and leaves the rest
    of each destination row untouched. ``src`` must be pinned host memory and ``dst``
    device memory.

    Use it only when ``width`` is actually smaller than the row: at width == pitch the
    driver's 2D path is ~7x slower than a plain contiguous copy (see memcpy2d.cuh).
    """
    assert src.is_contiguous() and dst.is_contiguous()
    assert src.shape == dst.shape and src.dtype == dst.dtype
    pitch = src.shape[-1] * src.element_size()
    assert 0 < width < pitch, f"memcpy2d wants a narrowing copy, got {width} of {pitch}"
    height = src.numel() // src.shape[-1]
    handle = stream if stream is not None else torch.cuda.current_stream().cuda_stream
    _jit_memcpy2d_module().memcpy2d(
        dst.data_ptr(), pitch, src.data_ptr(), pitch, width, height, handle
    )
