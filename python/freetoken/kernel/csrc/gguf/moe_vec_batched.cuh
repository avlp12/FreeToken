// FreeToken addition: weight-reuse batching for the grouped-expert GEMV.
//
// WHY
// ---
// ``moe_vec_q`` (moe_vec.cuh) gives ONE CUDA block to each (routed row, weight
// row) pair, and every block streams its expert's weight row in from HBM. At an
// 8192-token prefill chunk with top_k = 6 that is 49152 routed rows spread over
// 256 experts, so each expert's weight matrix is re-read ~192 times per layer:
// measured at 20.4 TB of traffic per chunk, ~69% of the card's HBM roofline and
// 89% of prefill wall time. The dot products themselves are a rounding error;
// the kernel is a weight-streaming loop wearing a GEMV costume.
//
// THE FIX
// -------
// ``vec_dot_*_q8_1(const void* vbq, const block_q8_1* bq8_1, const int& iqs)``
// takes the WEIGHT block pointer and the ACTIVATION block pointer as separate
// arguments. So a single thread can call it N times with the same ``vbq`` and N
// different ``bq8_1``. The weight block is loaded once and stays hot in L1
// across those N consecutive calls, cutting weight traffic ~N-fold, while each
// of the N outputs remains the exact same dot product accumulated in the exact
// same order as the unbatched kernel produced it.
//
// That last point is the whole reason this design was chosen over a real tiled
// GEMM: the batched path is BIT-IDENTICAL to the unbatched one, not merely
// close. Per-lane accumulation order, the warp reduction's shuffle sequence and
// the final store are unchanged; only the *scheduling* of which routed rows
// share a block differs. ``tests/kernels/test_moe_vec_batched.py`` asserts
// ``torch.equal``, not ``allclose``.
//
// GROUPING CONTRACT (host side)
// -----------------------------
// The N routed rows a block handles MUST all route to the SAME expert -- that
// is what makes one weight row serve all N. The host therefore passes ``perm``,
// an int32 array of routed-row indices sorted by expert and padded so that every
// aligned run of N entries belongs to one expert (the usual
// ``moe_align_block_size`` shape). Padding entries are -1 and are computed but
// never stored. ``perm`` is built once per ``fused_experts_q2k_ud`` call and
// serves BOTH of that call's GEMVs, since a routed-row index means the same
// thing in each.
//
// TYPE COVERAGE
// -------------
// Only the three ggml types the ``q2_k_ud`` expert banks actually hold are
// instantiated (Q2_K, IQ2_XS, IQ3_XXS). Every (type, N) pair is a separate
// kernel and this translation unit already compiles 19 unbatched ones; blanket
// instantiation would multiply JIT build time for launchers nothing calls.
// ``ggml_moe_vec_batched_supported`` reports the covered set so callers can fall
// back to the unbatched path instead of hitting a TORCH_CHECK.

#pragma once

// Batch widths we instantiate. Powers of two only: the host pads each expert's
// run up to a multiple of N, so a larger N costs more padding, and N is also the
// unroll factor for the accumulator/pointer arrays below.
#define MOE_VEC_BATCH_WIDTHS(F) F(2) F(4) F(8) F(16)

template <
    typename scalar_t,
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda,
    int N>
static __global__ void moe_vec_q_batched(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ topk_ids,
    const int* __restrict__ perm,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int64_t row_pitch_bytes,
    const int64_t z_offset) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;

  // One block now covers a GROUP of N routed rows instead of a single one.
  // gridDim.z is still capped at 65535, so the launcher slices the group range
  // exactly as the unbatched launcher slices the routed-row range.
  const int64_t group = z_offset + (int64_t)blockIdx.z;
  const int64_t base = group * (int64_t)N;

  // Routed-row indices this block owns; -1 marks alignment padding.
  int z[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    z[j] = perm[base + j];
  }

  // First non-padding lane. The host always packs real entries at the front of a
  // group, but nothing in the kernel depends on that; a wholly-padding group
  // (the tail of the fixed-size ``perm`` allocation) just exits.
  int lead = -1;
#pragma unroll
  for (int j = N - 1; j >= 0; --j) {
    if (z[j] >= 0) {
      lead = z[j];
    }
  }
  if (lead < 0) {
    return;
  }
  if (row >= nrows) {
    return;
  }

  // All N lanes share this expert -- that is the grouping contract, and it is
  // what lets us read the weight row ONCE for the whole group.
  const int expert = topk_ids[lead];

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // Byte pitch between consecutive expert rows; see moe_vec_resolve_pitch.
  const int64_t row_bytes =
      row_pitch_bytes > 0 ? row_pitch_bytes : (int64_t)blocks_per_row * (int64_t)sizeof(block_q_t);

  const block_q_t* x =
      (const block_q_t*)((const char*)vx + ((int64_t)expert * (int64_t)nrows + (int64_t)row) * row_bytes);

  // Activation row per lane. Padding lanes are pointed at the lead lane's token
  // so their (discarded) dot product still reads mapped memory -- cheaper and
  // safer than branching inside the hot loop.
  const block_q8_1* y[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const int64_t zj = z[j] >= 0 ? (int64_t)z[j] : (int64_t)lead;
    const int64_t token = zj / topk;
    y[j] = (const block_q8_1*)(((const int*)vy) + token * (int64_t)token_stride);
  }

  float tmp[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    tmp[j] = 0.0f;
  }

  // THE hot loop. ``&x[i]`` is identical across the inner j sweep, so the weight
  // block is fetched once and reused N times out of L1; ``y[j][iby]`` is the
  // only thing that varies. Each tmp[j] therefore accumulates the same terms in
  // the same order the unbatched kernel used for that routed row.
  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int iby = i * (qk / QK8_1);                  // y block index aligned with i
    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index
#pragma unroll
    for (int j = 0; j < N; ++j) {
      tmp[j] += vec_dot_q_cuda(&x[i], &y[j][iby], iqs);
    }
  }

  // N independent warp reductions, each running the identical mask sequence the
  // unbatched kernel runs.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      tmp[j] += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp[j], mask);
    }
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      // The padding guard. Without it a -1 lane writes at dst[-nrows + row],
      // i.e. straight through the front of the output allocation.
      if (z[j] >= 0) {
        dst[(int64_t)z[j] * (int64_t)nrows + (int64_t)row] = tmp[j];
      }
    }
  }
}

template <
    typename scalar_t,
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda,
    int N>
static void moe_vec_launch_batched(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int* perm,
    const int64_t top_k,
    const int64_t padded_total,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int64_t row_pitch_bytes,
    cudaStream_t stream) {
  TORCH_CHECK(top_k > 0 && top_k <= 2147483647LL, "moe_vec_batched: top_k out of range (", top_k, ")");
  TORCH_CHECK(
      padded_total % N == 0,
      "moe_vec_batched: perm length (",
      padded_total,
      ") is not a multiple of the batch width (",
      N,
      ")");
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  const int64_t pitch = moe_vec_resolve_pitch<block_q_t>(ncols / qk, row_pitch_bytes);
  const int64_t groups = padded_total / N;
  // Same gridDim.z slicing as the unbatched launcher -- the cap still bites,
  // just N times later (49152 routed rows at N = 8 is 6144 groups).
  for (int64_t z_offset = 0; z_offset < groups; z_offset += MOE_VEC_MAX_GRID_Z) {
    const int64_t remaining = groups - z_offset;
    const unsigned int z_span =
        (unsigned int)(remaining < MOE_VEC_MAX_GRID_Z ? remaining : (int64_t)MOE_VEC_MAX_GRID_Z);
    const dim3 block_nums(block_num_y, 1, z_span);
    moe_vec_q_batched<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, N>
        <<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, topk_ids, perm, (int)top_k, ncols, nrows, token_stride, pitch, z_offset);
  }
}

// One launcher per quant type, dispatching the runtime batch width onto the
// template parameter. Generated, for the same reason the unbatched launchers
// are: hand-copying the width switch per type is how divergence starts.
#define MOE_VEC_BATCHED_CASE(NV)                                                        \
  case NV:                                                                              \
    moe_vec_launch_batched<scalar_t, qk_v, qi_v, block_t, vdr_v, vec_dot_v, NV>(         \
        vx, vy, dst, topk_ids, perm, top_k, padded_total, ncols, nrows, token_stride,    \
        row_pitch_bytes, stream);                                                        \
    break;

#define MOE_VEC_BATCHED_LAUNCHER(name, qk, qi, block_q_t, vdr, vec_dot)                 \
  template <typename scalar_t>                                                          \
  static void name(                                                                     \
      const void* vx,                                                                   \
      const void* vy,                                                                   \
      scalar_t* dst,                                                                    \
      const int* topk_ids,                                                              \
      const int* perm,                                                                  \
      const int64_t top_k,                                                              \
      const int64_t padded_total,                                                       \
      const int ncols,                                                                  \
      const int nrows,                                                                  \
      const int token_stride,                                                           \
      const int64_t row_pitch_bytes,                                                    \
      const int64_t batch_n,                                                            \
      cudaStream_t stream) {                                                            \
    constexpr int qk_v = qk;                                                            \
    constexpr int qi_v = qi;                                                            \
    using block_t = block_q_t;                                                          \
    constexpr int vdr_v = vdr;                                                          \
    constexpr vec_dot_q_cuda_t vec_dot_v = vec_dot;                                     \
    switch (batch_n) {                                                                  \
      MOE_VEC_BATCH_WIDTHS(MOE_VEC_BATCHED_CASE)                                        \
      default:                                                                          \
        TORCH_CHECK(false, "moe_vec_batched: unsupported batch width ", batch_n);        \
    }                                                                                   \
  }

// The q2_k_ud bank types, and only those -- see TYPE COVERAGE above.
MOE_VEC_BATCHED_LAUNCHER(
    moe_vec_batched_q2_K_q8_1_cuda, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MOE_VEC_BATCHED_LAUNCHER(
    moe_vec_batched_iq2_xs_q8_1_cuda, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1)
MOE_VEC_BATCHED_LAUNCHER(
    moe_vec_batched_iq3_xxs_q8_1_cuda, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1)

// ggml type ids the batched path has kernels for. Q2_K = 10, IQ2_XS = 17,
// IQ3_XXS = 18.
static inline bool moe_vec_batched_has_type(int64_t type) {
  return type == 10 || type == 17 || type == 18;
}
