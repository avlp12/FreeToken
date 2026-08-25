"""Block-scaled FP8 (e4m3) linear for DeepSeek-V4, matching the reference numerics.

The reference (``inference/model.py`` ``linear`` + ``inference/kernel.py``
``act_quant``/``fp8_gemm``) quantizes the *activation* to FP8 with a per-128 block
power-of-two (ue8m0) scale, then runs an FP8xFP8 block-scaled GEMM against the FP8
weight (which carries its own 128x128 ue8m0 block scale). Both operands' scales are
applied per 128-K block to a separate FP32 accumulator. This module reproduces that:

  ``y = fp8_gemm(act_quant(x, 128, ue8m0), weight_fp8, weight_scale_e8m0)``

``act_quant`` (reference): per block ``s = 2**ceil(log2(max(|x|,1e-4)/448))`` (exact
via IEEE bit ops -> matches ``fast_round_scale``), ``x_fp8 = round_e4m3(clamp(x/s,
+-448))``, scale stored e8m0. The GEMM accumulates ``sum_k (A_fp8 @ B_fp8) * s_a * s_b``
per 128-K block in FP32.

Also provides ``act_quant_fp8_inplace`` -- the fused FP8 quant+dequant round-trip the
reference applies in-place to the window / compressor KV (``act_quant(..., 64, ...,
inplace=True)``), returning BF16.

Assumes ``K % 128 == 0`` and ``N % 128 == 0`` (true for every DeepSeek-V4 projection).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_act_dtype,
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    round_e4m3,
)

FP8 = torch.float8_e4m3fn
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# ``FREETOKEN_DSV4_FUSED_GEMV=0`` reverts the T=1 decode path to the pre-fusion three
# launches (act_quant -> gemv -> splitk reduce). The fused kernel is bit-identical at the
# shipped configs, so this is an escape hatch, not a numerics switch.
_USE_FUSED = os.environ.get("FREETOKEN_DSV4_FUSED_GEMV", "1").lower() not in (
    "0", "false", "no", "off",
)


# ======================================================================================
# Activation FP8 quantization (ue8m0 power-of-two scale), matching reference act_quant.
# ======================================================================================
@triton.jit
def _log2_ceil(v):
    """Exact ceil(log2(v)) for v > 0 via IEEE-754 bit ops (matches fast_log2_ceil)."""
    bits = v.to(tl.uint32, bitcast=True)
    exp = ((bits >> 23) & 0xFF).to(tl.int32)
    man = (bits & 0x7FFFFF).to(tl.int32)
    return exp - 127 + tl.where(man != 0, 1, 0)


@triton.jit
def _act_quant_fp8_kernel(
    x_ptr, y_ptr, s_ptr, M, N,
    stride_xm, stride_xn, stride_ym, stride_yn, stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """Per-row, per-``BLOCK`` FP8 quant with ue8m0 (pow2) scale. ``s`` holds e8m0 codes."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK + tl.arange(0, BLOCK)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-4)
    e = _log2_ceil(amax * (1.0 / 448.0))                # [BLOCK_M]
    s = tl.exp2(e.to(tl.float32))
    y = tl.clamp(x / s[:, None], -448.0, 448.0)
    if e4m3_native_cx():
        y = y.to(tl.float8e4nv)
    else:
        y = round_e4m3(y)  # e4m3-grid values into the wrapper's bf16 buffer
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yn,
        y, mask=m_mask[:, None],
    )
    code = (e + 127).to(tl.uint8)
    tl.store(s_ptr + offs_m * stride_sm + pid_n * stride_sn, code, mask=m_mask)


def act_quant_fp8(x: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``act_quant`` (ue8m0): returns ``(x_fp8 [M,K], scale_codes [M,K//block])``
    where ``scale = 2**(code-127)``. Without native fp8 the quantized values are the
    same e4m3-grid points held in bf16 (exactly representable)."""
    *lead, K = x.shape
    assert K % block == 0, (K, block)
    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    y = torch.empty((M, K), dtype=e4m3_act_dtype(), device=x.device)
    s = torch.empty((M, K // block), dtype=torch.uint8, device=x.device)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), K // block)
    _act_quant_fp8_kernel[grid](
        x2d, y, s, M, K,
        x2d.stride(0), x2d.stride(1), y.stride(0), y.stride(1), s.stride(0), s.stride(1),
        BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return y, s


@triton.jit
def _act_quant_inplace_kernel(
    x_ptr, o_ptr, M, N, stride_m, stride_n, stride_om, stride_on,
    FP8_MIN, FP8_MAX, INV_MAX, FP4: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused quant+dequant round-trip (reference ``inplace=True``), written to ``o_ptr`` as
    the input dtype (``o_ptr==x_ptr`` for true in-place; a distinct out buffer fuses the
    copy for callers that must not clobber the input). ``FP4=False`` -> FP8 e4m3 (block 64);
    ``FP4=True`` -> FP4 e2m1 (block 32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK + tl.arange(0, BLOCK)
    m_mask = offs_m < M
    ptrs = x_ptr + offs_m[:, None] * stride_m + offs_k[None, :] * stride_n
    x = tl.load(ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if FP4:
        amax = tl.maximum(amax, 6.0 * (2.0 ** -126))
    else:
        amax = tl.maximum(amax, 1e-4)
    e = _log2_ceil(amax * INV_MAX)
    s = tl.exp2(e.to(tl.float32))
    q = tl.clamp(x / s[:, None], FP8_MIN, FP8_MAX)
    if FP4:
        q = _round_fp4(q)
    elif e4m3_native_cx():
        q = q.to(tl.float8e4nv).to(tl.float32)
    else:
        q = round_e4m3(q)
    optrs = o_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    y = (q * s[:, None]).to(optrs.dtype.element_ty)
    tl.store(optrs, y, mask=m_mask[:, None])


@triton.jit
def _round_fp4(x):
    """Round to nearest float4_e2m1fn value in {0,.5,1,1.5,2,3,4,6} (signed), in FP32.

    Matches the hardware ``float4_e2m1fn`` cast: round-to-nearest, ties-to-even on the
    grid magnitudes. Even-magnitude grid points are {0, 1.0, 2.0, 4.0} (even mantissa
    bit), so the odd-magnitude midpoints (0.75, 1.75, 3.5) round UP to the even neighbor
    while the even-magnitude midpoints (0.25, 1.25, 2.5, 5.0) round toward the even one.
    Verified against the tilelang reference fp4 cast (probe: 0.75->1, 1.75->2, 3.5->4)."""
    sign = tl.where(x < 0, -1.0, 1.0)
    a = tl.abs(x)
    r = tl.where(
        a <= 0.25, 0.0,          # 0.25 tie -> 0.0 (even)
        tl.where(a < 0.75, 0.5,  # 0.75 tie -> 1.0 (even)
        tl.where(a <= 1.25, 1.0, # 1.25 tie -> 1.0 (even)
        tl.where(a < 1.75, 1.5,  # 1.75 tie -> 2.0 (even)
        tl.where(a <= 2.5, 2.0,  # 2.5 tie -> 2.0 (even)
        tl.where(a < 3.5, 3.0,   # 3.5 tie -> 4.0 (even)
        tl.where(a <= 5.0, 4.0, 6.0)))))))
    return sign * r


def act_quant_fp8_inplace(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Reference ``act_quant(x, block, ue8m0, e8m0, inplace=True)``: FP8 quant+dequant
    round-trip written back into ``x`` (BF16). Operates on the (possibly strided) view."""
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, x2d, M, N, x2d.stride(0), x2d.stride(1), x2d.stride(0), x2d.stride(1),
        -448.0, 448.0, 1.0 / 448.0, False, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return x


def act_quant_fp8_roundtrip(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 quant+dequant round-trip into a fresh contiguous BF16 tensor (fuses the copy --
    for callers that must keep ``x`` intact, e.g. the MoE expert input shared with the
    gate / shared expert). Numerically identical to ``act_quant_fp8_inplace(x.clone())``."""
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    out = torch.empty_like(x2d)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, out, M, N, x2d.stride(0), x2d.stride(1), out.stride(0), out.stride(1),
        -448.0, 448.0, 1.0 / 448.0, False, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return out.reshape(x.shape)


def fp4_act_quant_inplace(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Reference ``fp4_act_quant(x, block, inplace=True)``: FP4 quant+dequant round-trip
    written back into ``x`` (BF16)."""
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, x2d, M, N, x2d.stride(0), x2d.stride(1), x2d.stride(0), x2d.stride(1),
        -6.0, 6.0, 1.0 / 6.0, True, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return x


# ======================================================================================
# FP8 (act) x FP8 (weight) block-scaled GEMM / GEMV.
# ======================================================================================
@triton.jit
def _fp8_act_gemm_kernel(
    a_ptr,            # [M, K] float8_e4m3fn (quantized activation)
    w_ptr,            # [N, K] float8_e4m3fn
    sa_ptr,           # [M, K//128] uint8 (e8m0 act codes)
    sb_ptr,           # [N//128, K//128] uint8 (e8m0 weight codes)
    c_ptr,            # [M, N] compute dtype
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk,
    stride_sam, stride_sak, stride_sbn, stride_sbk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for k in range(num_k):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        if e4m3_native_cx():
            p = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        else:
            # bf16 dot on the same e4m3 grid: operands exact in bf16, fp32 acc
            p = tl.dot(a, tl.trans(e4m3_u8_to_f32(w).to(tl.bfloat16)), out_dtype=tl.float32)
        sa_code = tl.load(sa_ptr + offs_m * stride_sam + k * stride_sak, mask=m_mask, other=0)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)            # [BLOCK_M]
        sb_code = tl.load(sb_ptr + pid_n * stride_sbn + k * stride_sbk)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)            # scalar (one 128-N block)
        acc += p * sca[:, None] * scb
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None])


@triton.jit
def _fp8_act_gemv_splitk_kernel(
    a_ptr,            # [K] float8_e4m3fn
    sa_ptr,           # [K//128] uint8 (e8m0 act codes)
    w_ptr,            # [N, K] float8_e4m3fn
    sb_ptr,           # [N//128, K//128] uint8 (e8m0 weight codes)
    part_ptr,         # [SPLIT_K, N] fp32
    N, K,
    stride_ak, stride_wn, stride_wk, stride_sbn, stride_sbk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = offs_n // 128
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, k_per, 128):
        offs_k = k_start + k0 + tl.arange(0, 128)
        a = tl.load(a_ptr + offs_k * stride_ak).to(tl.float32)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=n_mask[:, None], other=0))
        kb = (k_start + k0) // 128
        sb_code = tl.load(sb_ptr + sn * stride_sbn + kb * stride_sbk, mask=n_mask, other=0)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)
        sa_code = tl.load(sa_ptr + kb)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)
        acc += tl.sum(w * a[None, :], axis=1) * scb * sca
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(part_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
                          stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(OUT), mask=mask)


# ======================================================================================
# Fused decode GEMV: act-quant in the prologue, split-k reduce in the epilogue.
#
# The unfused decode path is three launches per projection (``_act_quant_fp8_kernel`` ->
# ``_fp8_act_gemv_splitk_kernel`` -> ``_splitk_reduce_kernel``), and at T=1 the first and
# last are pure dispatch: ~1.0us each for a kernel that touches K bytes. On the DSV4
# decode step that is 322 projections x 3 = 966 graph nodes/token; the two bookend
# kernels alone burn ~0.62 ms/token of GPU time for ~0 arithmetic.
#
# The fused kernel folds both away:
#   * PROLOGUE -- every program quantizes the 128-wide activation slice it is about to
#     consume, in registers, with the *same* arithmetic as ``_act_quant_fp8_kernel``
#     (amax -> ue8m0 exponent via ``_log2_ceil`` -> clamp -> e4m3 round). The activation
#     row is at most 16 KB and every program reads the same bytes, so the redundant
#     re-quantization is served from L2 and costs no DRAM traffic. This makes the
#     activation operand BIT-IDENTICAL to the materialized one, by construction.
#   * EPILOGUE -- at ``SPLIT_K == 1`` the accumulator IS the result, so the program
#     stores it straight to ``out``; that is exactly what ``_splitk_reduce_kernel``
#     would have computed (``0.0 + part[0]``). At ``SPLIT_K > 1`` the k-partitions
#     still have to meet somewhere, so the last one to arrive at an n-tile does the
#     reduce itself (``_EP_LOCK``): each program stores its partial, then bumps
#     a per-tile arrival counter with an acq_rel GPU-scope atomic; the program that
#     gets ``SPLIT_K - 1`` back resets the counter and sums the SPLIT_K partials in
#     index order -- the SAME order, and therefore the same fp32 rounding, as
#     ``_splitk_reduce_kernel``. No program ever spins, so there is nothing to
#     deadlock, and the counter is left at 0 for the next replay.
#
# Both epilogues are bit-identical to the unfused path at a matching (BLOCK_N, SPLIT_K),
# not merely close -- ``tests/kernels/test_dsv4_fp8_fused_gemv.py`` pins that.
# ======================================================================================
_EP_DIRECT = 0   # SPLIT_K == 1: store the accumulator as the result
_EP_PART = 1     # write partials; caller runs _splitk_reduce_kernel (parity reference)
_EP_LOCK = 2     # write partials; the last arrival at the tile reduces them


@triton.jit
def _fp8_fused_gemv_kernel(
    x_ptr,            # [G, K] UNQUANTIZED activation (bf16/fp16/fp32)
    w_ptr,            # [N, K] float8_e4m3fn (or its uint8 view)
    sb_ptr,           # [N//128, K//128] uint8 (e8m0 weight codes)
    o_ptr,            # _EP_DIRECT: [N] out dtype; else [SPLIT_K, N] fp32 partials
    out_ptr,          # _EP_LOCK: [N] out dtype (unused otherwise)
    lock_ptr,         # _EP_LOCK: [n_tiles] int32 arrival counters, zero on entry+exit
    N, K,
    stride_xg, stride_xk, stride_wn, stride_wk, stride_sbn, stride_sbk,
    stride_ok, stride_on,
    GROUP_ROWS: tl.constexpr,   # output rows fed by one activation row (N when G == 1)
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    OUT: tl.constexpr, EPILOGUE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = offs_n // 128
    # Grouped (block-diagonal) form: output rows [g*GROUP_ROWS, (g+1)*GROUP_ROWS) read
    # activation row g. BLOCK_N divides GROUP_ROWS, so a tile never straddles two groups
    # and the group index is a scalar.
    xg = x_ptr + (pid_n * BLOCK_N // GROUP_ROWS) * stride_xg
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, k_per, 128):
        offs_k = k_start + k0 + tl.arange(0, 128)
        # --- fused act_quant, bit-identical to _act_quant_fp8_kernel ---
        xv = tl.load(xg + offs_k * stride_xk).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xv)), 1e-4)
        e = _log2_ceil(amax * (1.0 / 448.0))
        sca = tl.exp2(e.to(tl.float32))
        q = tl.clamp(xv / sca, -448.0, 448.0)
        if e4m3_native_cx():
            a = q.to(tl.float8e4nv).to(tl.float32)
        else:
            a = round_e4m3(q)
        # --- the GEMV body, unchanged ---
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=n_mask[:, None], other=0))
        kb = (k_start + k0) // 128
        sb_code = tl.load(sb_ptr + sn * stride_sbn + kb * stride_sbk, mask=n_mask, other=0)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)
        acc += tl.sum(w * a[None, :], axis=1) * scb * sca
    if EPILOGUE == 0:      # _EP_DIRECT
        tl.store(o_ptr + offs_n * stride_on, acc.to(OUT), mask=n_mask)
    elif EPILOGUE == 1:    # _EP_PART
        tl.store(o_ptr + pid_k * stride_ok + offs_n * stride_on, acc, mask=n_mask)
    else:                  # _EP_LOCK -- last k-partition of this tile reduces in place
        p_ptrs = o_ptr + pid_k * stride_ok + offs_n * stride_on
        tl.store(p_ptrs, acc, mask=n_mask, cache_modifier=".cg")
        tl.debug_barrier()  # every thread's partial is written before we announce arrival
        arrivals = tl.atomic_add(lock_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrivals == SPLIT_K - 1:
            # Leave the counter at 0 so the buffer is reusable (and CUDA-graph replayable)
            # with no host-side reset.
            tl.atomic_xchg(lock_ptr + pid_n, 0, sem="relaxed", scope="gpu")
            tot = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                tot += tl.load(o_ptr + k * stride_ok + offs_n * stride_on,
                               mask=n_mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + offs_n, tot.to(OUT), mask=n_mask)


# Swept-best FUSED decode GEMV config per (N, K) -> (BLOCK_N, SPLIT_K, num_warps,
# num_stages), from ``benchmarks/bench_dsv4_fp8_gemv.py --sweep`` on an RTX 5090
# (170 SMs, 128 MB L2, 1.79 TB/s), measured DRAM-bound inside a CUDA graph.
#
# The fused epilogue makes SPLIT_K free of launch cost, so these run far narrower tiles
# (BLOCK_N 2-8) and more k-partitions than the unfused table could afford: with the
# reduce as a separate launch, every extra k-partition had to pay for itself twice over.
# Narrow tiles are what buys the bandwidth here -- one token's GEMV has only N/BLOCK_N
# x SPLIT_K programs to fill 170 SMs with memory-level parallelism.
_FUSED_DECODE_CFG: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (1024, 4096): (4, 8, 1, 1),     # wq_a
    (32768, 1024): (32, 1, 1, 3),   # wq_b
    (512, 4096): (2, 8, 1, 3),      # attn wkv
    (4096, 8192): (8, 4, 1, 3),     # wo_b
    (2048, 4096): (4, 4, 1, 3),     # shared w1 / w3
    (4096, 2048): (4, 2, 1, 1),     # shared w2
    (8192, 1024): (4, 1, 1, 3),     # indexer wq_b
    (8192, 4096): (8, 2, 1, 1),     # wo_a, grouped (FREETOKEN_WO_A_FP8)
}


def _fused_cfg(N: int, K: int) -> tuple[int, int, int, int]:
    cfg = _FUSED_DECODE_CFG.get((N, K))
    if cfg is not None:
        bn, sk, nw, ns = cfg
    else:
        bn, sk, nw = _decode_cfg(N, K)
        ns = 3
    sk = max(1, min(sk, K // 128))
    while sk > 1 and (K // sk) % 128:  # every k-slice must stay a whole number of blocks
        sk //= 2
    return bn, sk, nw, ns


# Per-device arrival counters for the _EP_LOCK epilogue. One int32 per n-tile; the
# kernel leaves every entry back at 0, so ONE buffer serves every projection and survives
# CUDA-graph replay untouched (nothing has to reset it between calls). Sized once for the
# widest decode grid with headroom, so it is allocated exactly once per device.
#
# The buffer is shared across call sites, which is safe because the decode step issues its
# projections on ONE stream: kernels are serialized, so no two GEMVs ever hold the counters
# at the same time. Two GEMVs on concurrent streams would race, hence the single-stream
# assumption is load-bearing.
_LOCK_SLOTS = 1 << 14
_LOCKS: dict[torch.device, torch.Tensor] = {}


def _locks(device: torch.device, n_tiles: int) -> torch.Tensor | None:
    """The arrival-counter buffer, or None if it would have to be allocated mid-capture
    (the caller then takes the two-launch epilogue, which needs no counters)."""
    buf = _LOCKS.get(device)
    if buf is None or buf.numel() < n_tiles:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = torch.zeros(max(n_tiles, _LOCK_SLOTS), dtype=torch.int32, device=device)
        _LOCKS[device] = buf
    return buf


def fused_fp8_gemv(
    x: torch.Tensor,            # [G, K] or [K] unquantized activation
    weight: torch.Tensor,       # [N, K] float8_e4m3fn (kernel view already applied)
    sb: torch.Tensor,           # [N//128, K//128] uint8 e8m0
    out_dtype: torch.dtype,
    *,
    group_rows: int | None = None,
    cfg: tuple[int, int, int, int] | None = None,
    epilogue: int | None = None,
) -> torch.Tensor:
    """Single-launch FP8 block-scaled GEMV: activation quantized in the prologue,
    split-k reduced in the epilogue. ``group_rows`` selects the block-diagonal (grouped)
    form used by ``wo_a`` -- output rows ``[g*group_rows, (g+1)*group_rows)`` consume
    activation row ``g``. ``epilogue`` forces ``_EP_PART`` (two launches) for parity
    testing; the default picks ``_EP_DIRECT`` / ``_EP_LOCK`` by ``SPLIT_K``."""
    N, K = weight.shape
    x2 = x.reshape(-1, K)
    G = x2.shape[0]
    gr = N if group_rows is None else group_rows
    assert gr * G == N, (gr, G, N)
    BLOCK_N, split_k, num_warps, num_stages = cfg or _fused_cfg(N, K)
    assert gr % BLOCK_N == 0, (gr, BLOCK_N)
    n_tiles = triton.cdiv(N, BLOCK_N)
    ep = (_EP_DIRECT if split_k == 1 else _EP_LOCK) if epilogue is None else epilogue
    out = torch.empty(N, dtype=out_dtype, device=x.device)
    if ep == _EP_DIRECT:
        assert split_k == 1
        o, lock = out, out          # unused pointer args; the branch is constexpr-dead
        stride_ok, stride_on = 0, out.stride(0)
    else:
        lock = out
        if ep == _EP_LOCK:
            lock = _locks(x.device, n_tiles)
            if lock is None:  # counters not warmed up before capture -> two launches
                ep, lock = _EP_PART, out
        o = torch.empty((split_k, N), dtype=torch.float32, device=x.device)
        stride_ok, stride_on = o.stride(0), o.stride(1)
    _fp8_fused_gemv_kernel[(n_tiles, split_k)](
        x2, weight, sb, o, out, lock, N, K,
        x2.stride(0), x2.stride(1), weight.stride(0), weight.stride(1),
        sb.stride(0), sb.stride(1), stride_ok, stride_on,
        GROUP_ROWS=gr, BLOCK_N=BLOCK_N, SPLIT_K=split_k,
        OUT=_TL_DTYPE[out_dtype], EPILOGUE=ep,
        num_warps=num_warps, num_stages=num_stages,
    )
    if ep == _EP_PART:
        _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
            o, out, N, split_k, o.stride(0), o.stride(1),
            BLOCK=256, OUT=_TL_DTYPE[out_dtype], num_warps=2,
        )
    return out


def grouped_block_fp8_linear(
    x: torch.Tensor,          # [..., G, D] bf16 (one activation row per output group)
    weight: torch.Tensor,     # [G*R, D] float8_e4m3fn
    scale: torch.Tensor,      # [G*R//128, D//128] e8m0
    group_rows: int,          # R
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-diagonal FP8 linear: ``out[g*R + r] = sum_d x[g, d] * W[g*R + r, d]``.

    This is the FP8 form of DSV4's ``wo_a`` grouped output projection (the reference
    runs it as a bf16 ``einsum("bsgd,grd->bsgr")``). Decode (one token) is a single
    fused GEMV launch; prefill falls back to one block GEMM per group.
    """
    assert weight.dtype == FP8
    *lead, G, D = x.shape
    NR, R = weight.shape[0], group_rows
    assert weight.shape[1] == D and R * G == NR, (weight.shape, G, R)
    compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
    sb = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    sb = sb.contiguous()
    M = 1
    for s in lead:
        M *= s
    if M == 1:
        out = fused_fp8_gemv(
            x, e4m3_kernel_view(weight), sb, compute_dtype, group_rows=R
        ).reshape(*lead, NR)
    else:
        xm = x.reshape(M, G, D)
        out = torch.cat(
            [block_fp8_linear(xm[:, g], weight[g * R:(g + 1) * R],
                              sb[g * R // 128:(g + 1) * R // 128])
             for g in range(G)],
            dim=-1,
        ).reshape(*lead, NR)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# Swept-best decode GEMV config per (N, K) -> (BLOCK_N, SPLIT_K, num_warps).
_DECODE_FP8_CFG = {
    (1024, 4096): (16, 16, 1),   # wq_a
    (32768, 1024): (16, 1, 2),   # wq_b
    (512, 4096): (16, 16, 1),    # attn wkv
    (4096, 8192): (16, 8, 1),    # wo_b
    (2048, 4096): (16, 8, 1),    # shared w1 / w3
    (4096, 2048): (16, 8, 1),    # shared w2
    (8192, 1024): (16, 2, 4),    # indexer wq_b
}


def _decode_cfg(N: int, K: int) -> tuple[int, int, int]:
    cfg = _DECODE_FP8_CFG.get((N, K))
    if cfg is not None:
        bn, sk, nw = cfg
        return bn, max(1, min(sk, K // 128)), nw
    bn = 16
    n_tiles = triton.cdiv(N, bn)
    sk = max(1, 1536 // n_tiles)
    sk = 1 << (sk.bit_length() - 1)
    return bn, max(1, min(sk, K // 128)), 1


def _fp8_act_gemv(a_fp8: torch.Tensor, sa: torch.Tensor, weight: torch.Tensor,
                  sb: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    N, K = weight.shape
    BLOCK_N, split_k, num_warps = _decode_cfg(N, K)
    n_tiles = triton.cdiv(N, BLOCK_N)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a_fp8.device)
    _fp8_act_gemv_splitk_kernel[(n_tiles, split_k)](
        a_fp8, sa, weight, sb, part, N, K,
        a_fp8.stride(0), weight.stride(0), weight.stride(1),
        sb.stride(0), sb.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, SPLIT_K=split_k, num_warps=num_warps,
    )
    out = torch.empty(N, dtype=out_dtype, device=a_fp8.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype], num_warps=2,
    )
    return out


def block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    fused: bool | None = None,
) -> torch.Tensor:
    """``y = act_quant(x) @ weight^T`` (reference FP8 path).

    ``x``: ``[..., K]`` bf16; ``weight``: ``[N, K]`` float8_e4m3fn; ``scale``:
    ``[N//128, K//128]`` float8_e8m0fnu (weight block scale). Activation is quantized
    to FP8 with a per-128 ue8m0 scale; the GEMM applies both scales per 128-K block.

    ``fused`` overrides the decode-path kernel choice (default: the module toggle, i.e.
    the fused single-launch GEMV unless ``FREETOKEN_DSV4_FUSED_GEMV=0``). ``fused=False``
    is the pre-fusion three-launch path, kept as the parity/benchmark reference.
    """
    assert weight.dtype == FP8
    *lead, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert K % 128 == 0 and N % 128 == 0, (N, K)
    compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
    sb = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    sb = sb.contiguous()
    w = e4m3_kernel_view(weight)

    M = 1
    for s in lead:
        M *= s

    if M == 1:
        # Decode: one fused launch (act-quant prologue, split-k epilogue) instead of
        # act_quant -> gemv -> reduce. See _fp8_fused_gemv_kernel.
        if _USE_FUSED if fused is None else fused:
            out = fused_fp8_gemv(x, w, sb, compute_dtype).reshape(*lead, N)
            if bias is not None:
                out = out + bias.to(out.dtype)
            return out
        a_fp8, sa = act_quant_fp8(x, 128)
        out = _fp8_act_gemv(a_fp8[0], sa[0], w, sb, compute_dtype).reshape(*lead, N)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    a_fp8, sa = act_quant_fp8(x, 128)  # [M,K] fp8, [M,K//128] e8m0 codes

    out = torch.empty((M, N), dtype=compute_dtype, device=x.device)
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_act_gemm_kernel[grid](
        a_fp8, w, sa, sb, out,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1), w.stride(0), w.stride(1),
        sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        compute_type=_TL_DTYPE[compute_dtype], num_warps=4, num_stages=3,
    )
    out = out.reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = [
    "block_fp8_linear",
    "grouped_block_fp8_linear",
    "fused_fp8_gemv",
    "act_quant_fp8",
    "act_quant_fp8_inplace",
    "fp4_act_quant_inplace",
]
