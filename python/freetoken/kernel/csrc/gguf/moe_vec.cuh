// copied from
// https://github.com/vllm-project/vllm/blob/4492e3a55428e161ca8db381edc28263e5da4c8d/csrc/quantization/gguf/moe_vec.cuh
// copied and adapted from
// https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu

// FreeToken addition: resolve the byte PITCH between consecutive expert rows in
// the packed weight bank.
//
// The vendored kernel assumes rows sit back-to-back at their native packed width
// (`blocks_per_row * sizeof(block_q_t)`). FreeToken's mixed-quant expert bank
// instead gives every layer the same uniform row pitch, chosen from the WIDEST
// type in the bank, so a layer whose native row is narrower gets trailing
// padding. That pitch is a whole number of blocks of the widest type, which is
// in general NOT a whole number of blocks of the narrower type sharing the bank
// -- the production gate_up bank is 1568 B/row (16 IQ3_XXS blocks at H=4096)
// while 42 of its layers hold IQ2_XS rows of 74 B blocks, and 1568 % 74 != 0.
// The pitch is therefore BYTE-granular: it positions the row, and blocks tile
// from the row base exactly as they do in a tight bank.
//
// `row_pitch_bytes == 0` keeps the historical tightly-packed semantics
// bit-for-bit. A nonzero pitch must be at least the native packed row and a
// multiple of 16 B. The 16 B rule (not the block size) is what the hardware
// actually needs: the torch caching allocator hands out 256 B-aligned bases, so
// a 16 B-multiple pitch keeps every row base 16 B-aligned, which covers any
// int4/int2 vector load and is never weaker than the alignment a tight bank
// gives the same row. Both production pitches (1568, 784) satisfy it.
template <typename block_q_t>
static inline int64_t moe_vec_resolve_pitch(int blocks_per_row, int64_t row_pitch_bytes) {
  const int64_t native = (int64_t)blocks_per_row * (int64_t)sizeof(block_q_t);
  if (row_pitch_bytes <= 0) {
    return native;
  }
  TORCH_CHECK(
      row_pitch_bytes >= native,
      "moe_vec: row_pitch_bytes (",
      row_pitch_bytes,
      ") is narrower than the native packed row (",
      native,
      " bytes)");
  TORCH_CHECK(
      row_pitch_bytes % 16 == 0,
      "moe_vec: row_pitch_bytes (",
      row_pitch_bytes,
      ") must be a multiple of 16 bytes so every expert row starts 16 B-aligned");
  return row_pitch_bytes;
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* topk_ids,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int64_t row_pitch_bytes,
    const int64_t z_offset) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;

  // Routed-row index within the FULL [tokens * topk] range. gridDim.z is capped
  // at 65535 by the hardware, so the launcher slices that range and hands each
  // launch the base of its own chunk; blockIdx.z is the offset inside the chunk.
  // int64 throughout: at top_k = 6 a 16 K-token prefill chunk is already ~98 K
  // routed rows, and dst below is indexed by z * nrows.
  const int64_t z = z_offset + (int64_t)blockIdx.z;

  const int64_t token = z / topk;
  const auto expert = (topk_ids)[z];

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // Row-to-row stride in BYTES. Equals the native packed row when the bank is
  // tightly packed; wider when rows are padded to a uniform bank pitch, and not
  // necessarily a whole number of blocks of *this* type. Only ADDRESSING uses
  // it: the row base is computed in bytes, then blocks tile from that base as
  // they do in a tight bank, and the dequant/dot loop still runs blocks_per_row
  // iterations -- so the padding bytes are never read.
  const int64_t row_bytes =
      row_pitch_bytes > 0 ? row_pitch_bytes : (int64_t)blocks_per_row * (int64_t)sizeof(block_q_t);

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)((const char*)vx +
                                          ((int64_t)expert * (int64_t)nrows + (int64_t)row) * row_bytes);
  const block_q8_1* y = (const block_q8_1*)(((const int*)vy) + token * (int64_t)token_stride);

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    // x block index, now relative to this row's base rather than the expert's

    const int iby = i * (qk / QK8_1);  // y block index that aligns with i

    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int

    tmp += vec_dot_q_cuda(&x[i], &y[iby], iqs);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp, mask);
  }

  if (threadIdx.x == 0) {
    dst[z * (int64_t)nrows + (int64_t)row] = tmp;
  }
}

// The routed-row count `tokens * top_k` rides on gridDim.z, and the hardware
// caps gridDim.y/z at 65535 (only gridDim.x reaches 2^31-1). It has to stay on
// z: blockIdx.x is the fastest-varying axis and consecutive x-blocks are
// consecutive weight rows of the SAME (token, expert) pair, which is what keeps
// one expert's rows co-resident in L2 for this bandwidth-bound GEMV. Moving the
// row count to x would scatter the weight streaming.
//
// So past the cap we SLICE z instead, handing each launch the base of its own
// chunk in `z_offset` -- the same shape as `quantize_row_q8_1_cuda`'s loop over
// MAX_BLOCK_SIZE in gguf_kernel.cu. The offset is an explicit kernel argument
// rather than a base-pointer bump because the kernel derives `token = z / topk`
// from the absolute z; a pointer scheme would also force every chunk stride to
// be a multiple of topk.
//
// At top_k = 6 the old cap bit at 10923 tokens per prefill chunk
// (10923 * 6 = 65538 > 65535), which is why --max-prefill-length 16384 died on
// its first chunk with cudaErrorInvalidValue.
#define MOE_VEC_MAX_GRID_Z 65535

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void moe_vec_launch(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int64_t top_k,
    const int64_t tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int64_t row_pitch_bytes,
    cudaStream_t stream) {
  TORCH_CHECK(top_k > 0 && top_k <= 2147483647LL, "moe_vec: top_k out of range (", top_k, ")");
  TORCH_CHECK(tokens >= 0, "moe_vec: negative token count (", tokens, ")");
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  const int64_t pitch = moe_vec_resolve_pitch<block_q_t>(ncols / qk, row_pitch_bytes);
  const int64_t routed_rows = tokens * top_k;
  for (int64_t z_offset = 0; z_offset < routed_rows; z_offset += MOE_VEC_MAX_GRID_Z) {
    const int64_t remaining = routed_rows - z_offset;
    const unsigned int z_span =
        (unsigned int)(remaining < MOE_VEC_MAX_GRID_Z ? remaining : (int64_t)MOE_VEC_MAX_GRID_Z);
    const dim3 block_nums(block_num_y, 1, z_span);
    moe_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, dst, topk_ids, (int)top_k, ncols, nrows, token_stride, pitch, z_offset);
  }
}

// One launcher per quant type. They differ only in their template arguments, so
// they are generated rather than hand-written: 19 hand-copied bodies is exactly
// how the gridDim.z cap came to be replicated 19 times in the first place.
#define MOE_VEC_LAUNCHER(name, qk, qi, block_q_t, vdr, vec_dot)                     \
  template <typename scalar_t>                                                      \
  static void name(                                                                 \
      const void* vx,                                                               \
      const void* vy,                                                               \
      scalar_t* dst,                                                                \
      const int* topk_ids,                                                          \
      const int64_t top_k,                                                          \
      const int64_t tokens,                                                         \
      const int ncols,                                                              \
      const int nrows,                                                              \
      const int token_stride,                                                       \
      const int64_t row_pitch_bytes,                                                \
      cudaStream_t stream) {                                                        \
    moe_vec_launch<scalar_t, qk, qi, block_q_t, vdr, vec_dot>(                      \
        vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride,           \
        row_pitch_bytes, stream);                                                   \
  }

MOE_VEC_LAUNCHER(moe_vec_q4_0_q8_1_cuda, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q4_1_q8_1_cuda, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q5_0_q8_1_cuda, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q5_1_q8_1_cuda, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q8_0_q8_1_cuda, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q2_K_q8_1_cuda, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q3_K_q8_1_cuda, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q4_K_q8_1_cuda, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q5_K_q8_1_cuda, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
MOE_VEC_LAUNCHER(moe_vec_q6_K_q8_1_cuda, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq2_xxs_q8_1_cuda, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq2_xs_q8_1_cuda, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq2_s_q8_1_cuda, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq3_xxs_q8_1_cuda, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq1_s_q8_1_cuda, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq1_m_q8_1_cuda, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq4_nl_q8_1_cuda, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq4_xs_q8_1_cuda, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1)
MOE_VEC_LAUNCHER(moe_vec_iq3_s_q8_1_cuda, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1)
