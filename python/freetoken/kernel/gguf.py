"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

def _sources_digest() -> str:
    """Content hash of every source file in ``csrc/gguf/``.

    ``torch.utils.cpp_extension.load`` decides whether to rebuild by hashing the
    files named in ``sources`` plus the build flags -- and ``sources`` is just
    ``gguf_kernel.cu``. Edit any of the ``.cuh`` headers it pulls in and that hash
    is unchanged, so ``load`` short-circuits and silently hands back the STALE
    ``.so``: the kernel you are editing is not the kernel that runs, and nothing
    says so. (ninja tracks the header deps correctly via ``-MD``; it just never
    gets asked.)

    Folding a digest of the directory into an otherwise-dead ``-D`` puts the
    headers back inside the rebuild key, so a header edit rebuilds exactly once
    and an untouched tree still skips the build.
    """
    h = hashlib.sha256()
    for f in sorted(_CSRC.iterdir()):
        if f.suffix in (".cu", ".cuh", ".h"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    # ``-std=c++20`` is load-bearing, not a modernization: nvcc's host pass rewrites
    # ``static_cast<typename decltype(impl_->list)::difference_type>`` in libtorch's
    # ``ATen/core/List_inl.h`` into a form that drops the ``typename``, which every
    # g++ we have (12/13/15) rejects under C++17. C++20 (P0634) makes ``typename``
    # implicit in a static_cast type-id, so the rewritten form compiles. torch only
    # appends its own ``-std=c++17`` when no ``-std=`` is present, so this wins.
    extra_cuda_cflags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "-std=c++20",
        # Unused by the code; present only to pull the .cuh headers into
        # load()'s rebuild key. See _sources_digest.
        f"-DFREETOKEN_GGUF_SRC_DIGEST=g{_sources_digest()}",
    ]
    host_cxx = _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
    row_pitch_bytes: int = 0,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``.

    The MMQ path is *not* pitch-aware -- ``row_pitch_bytes`` must stay 0 (rows
    tightly packed at their native width). Padded-row banks go through
    :func:`ggml_moe_a8_vec`.
    """
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens, row_pitch_bytes,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    row_pitch_bytes: int = 0,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``.

    ``row_pitch_bytes`` is the byte distance between consecutive expert rows in
    ``weight``. The default 0 means "rows are tightly packed at their native
    width" -- identical addressing to the upstream kernel. Pass a nonzero pitch
    when the rows live in a bank padded to a uniform stride shared by several
    quant types. It is byte-granular and need NOT be a whole number of
    ``quant_type`` blocks -- a bank pitch sized for the widest type in the bank
    generally is not, e.g. 1568 B rows holding 74 B IQ2_XS blocks. It only has to
    be at least the native packed row and a multiple of 16 bytes, which keeps
    every row base 16 B-aligned.
    """
    return _module().ggml_moe_a8_vec(
        x, weight, topk_ids, top_k, quant_type, row, tokens, row_pitch_bytes
    )


def ggml_moe_vec_batched_supported(quant_type: int) -> bool:
    """Whether :func:`ggml_moe_a8_vec_batched` has a kernel for ``quant_type``.

    Only the ggml types the ``q2_k_ud`` expert banks hold are instantiated
    (Q2_K = 10, IQ2_XS = 17, IQ3_XXS = 18): every (type, batch width) pair is a
    separate kernel, and blanket instantiation across all 19 types would multiply
    JIT build time for launchers nothing calls. Callers check this and fall back
    to :func:`ggml_moe_a8_vec` rather than eating a hard error.
    """
    return bool(_module().ggml_moe_vec_batched_supported(quant_type))


def ggml_moe_a8_vec_batched(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    perm: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    row_pitch_bytes: int = 0,
    batch_n: int = 8,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight-reuse batched sibling of :func:`ggml_moe_a8_vec`.

    Same arguments, same semantics, same output BITS. The difference is purely how
    work is scheduled: ``batch_n`` routed rows that share an expert are handled by
    one CUDA block, so the expert's weight row crosses the memory bus once per
    ``batch_n`` rows instead of once per row. Each output element is still the
    identical dot product accumulated in the identical order, which is why the
    tests assert ``torch.equal`` and not ``allclose``.

    ``perm`` is an int32 CUDA tensor of routed-row indices grouped so that every
    aligned run of ``batch_n`` entries belongs to a single expert, with -1 in the
    alignment padding (see ``freetoken.moe.fused_q2_k_ud._expert_group_perm``). Its
    length must be a multiple of ``batch_n`` and may exceed ``tokens * top_k``:
    the builder sizes it to a fixed worst-case bound so it never needs a host
    sync, and the surplus -1 groups exit immediately.

    ``out``, when given, is written in place. Only real routed rows are ever
    stored to, so a poisoned ``out`` proves the padding lanes stayed silent.

    Not every quant type has a batched kernel -- ask
    :func:`ggml_moe_vec_batched_supported` first.
    """
    return _module().ggml_moe_a8_vec_batched(
        x, weight, topk_ids, perm, top_k, quant_type, row, tokens,
        row_pitch_bytes, batch_n, out,
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


def ggml_type_block_bytes(quant_type: int) -> int:
    """``sizeof(block_q_t)`` for a ggml type -- the byte width of ONE super-block.

    ``blocks_per_row * ggml_type_block_bytes(t)`` is the row width a natively
    packed bank of ``t`` occupies, which is exactly what :func:`ggml_dequantize`
    assumes (it walks ``m * n`` elements with no pitch parameter). The prefill
    dequant-GEMM path uses it to slice the native prefix out of a padded-pitch
    expert bank.

    NOT interchangeable with :func:`ggml_moe_get_block_size`, which returns the
    MMQ tile height ``MOE_X_*`` -- a row COUNT, 0 for every IQ type and 4 for
    Q2_K. Returns 0 for a type with no dequant kernel.
    """
    return _module().ggml_type_block_bytes(quant_type)


def ggml_type_block_elems(quant_type: int) -> int:
    """Elements per super-block (ggml's ``blck_size``) for a type; 0 if unknown.

    Companion to :func:`ggml_type_block_bytes` -- together they give
    ``native_row_bytes = ncols // block_elems * block_bytes``.
    """
    return _module().ggml_type_block_elems(quant_type)


def ggml_type_row_bytes(quant_type: int, ncols: int) -> int:
    """Natively packed byte width of one ``ncols``-element row of ``quant_type``.

    0 when the type is unknown to the dequant kernels. Raises when ``ncols`` is
    not a whole number of super-blocks -- a row that does not tile is not a row
    this path can slice.
    """
    elems = ggml_type_block_elems(quant_type)
    nbytes = ggml_type_block_bytes(quant_type)
    if elems == 0 or nbytes == 0:
        return 0
    if ncols % elems:
        raise ValueError(
            f"ncols {ncols} is not a multiple of the {elems}-element super-block "
            f"of ggml type {quant_type}"
        )
    return ncols // elems * nbytes


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_a8_vec_batched",
    "ggml_moe_vec_batched_supported",
    "ggml_moe_get_block_size",
    "ggml_type_block_bytes",
    "ggml_type_block_elems",
    "ggml_type_row_bytes",
]
