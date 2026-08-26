// Adatped from
// https://github.com/vllm-project/vllm/blob/755ed7b05be4743237d3339c4ff8c22bcaae04f4/csrc/quantization/gguf/gguf_kernel.cu
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

// dont use clang-format here, it breaks the include order
// clang-format off
#include "dispatch.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
#include "moe_vec_batched.cuh"
// clang-format off

// Q8 gemv
template <typename scalar_t>
static __global__ void
quantize_q8_1(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx, const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx, const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[(int64_t)off * (int64_t)kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

torch::Tensor ggml_dequantize(
    torch::Tensor W,  // quant weight
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      default:
        // Y was torch::empty'd above and no case ran: without this check an
        // unmatched type silently returns UNINITIALIZED memory as the GEMV
        // result instead of failing. See the module-level hazard note in
        // ggml_moe_a8_vec below.
        TORCH_CHECK(false, "ggml_mul_mat_vec_a8: unsupported ggml type ", type);
    }
  });
  return Y;
}

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      default:
        // Same hazard as ggml_mul_mat_vec_a8 above: Y is pre-allocated and an
        // unmatched type would silently return it uninitialized.
        TORCH_CHECK(false, "ggml_mul_mat_a8: unsupported ggml type ", type);
    }
  });
  return Y;
}

// NOTE: the MMQ (batched/prefill) grouped-expert path is NOT pitch-aware.
// Its expert stride already comes from ``W.stride(0)``, but the per-row stride is
// derived inside ``moe_q`` (moe.cuh) from ``ncols_x / qk`` and handed to the
// ``load_tiles_*`` helpers that mmq.cuh's plain path shares. Padded-row banks must
// therefore go through ``ggml_moe_a8_vec``; a nonzero pitch is rejected here
// rather than silently mis-addressed.
torch::Tensor ggml_moe_a8(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens,
    int64_t row_pitch_bytes) {
  TORCH_CHECK(
      row_pitch_bytes == 0,
      "ggml_moe_a8 (MMQ path) does not support a padded row pitch; got row_pitch_bytes=",
      row_pitch_bytes,
      ". Use ggml_moe_a8_vec for banks with padded rows.");
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      default:
        // Same hazard as ggml_mul_mat_vec_a8 above: Y is pre-allocated and an
        // unmatched type would silently return it uninitialized.
        TORCH_CHECK(false, "ggml_moe_a8: unsupported ggml type ", type);
    }
  });
  return Y;
}

// ``row_pitch_bytes`` is the distance in bytes between consecutive expert rows in
// ``W``. 0 (the default, and what every pre-existing call site passes implicitly)
// means "rows are tightly packed at their native width", i.e. exactly the
// upstream behaviour. A nonzero pitch lets the weight live in a bank whose rows
// are padded out to a uniform stride shared across quant types. It is
// BYTE-granular -- a bank pitch sized in blocks of the widest type in the bank is
// generally not a whole number of blocks of a narrower type sharing that bank --
// and only has to be >= the native row and a multiple of 16 B, which keeps every
// row base 16 B-aligned (see moe_vec_resolve_pitch).
torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens,
    int64_t row_pitch_bytes) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      // FreeToken: MXFP4, carried natively for the layers unsloth quantised at
      // 4.25 bpw instead of re-encoding them into the bank's narrower type.
      case 39:
        moe_vec_mxfp4_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            stream);
        break;
      default:
        // Y was torch::zeros'd above and no case ran: without this check an
        // unmatched type silently returns an ALL-ZERO GEMV result -- indistinguishable
        // from a correctly-computed answer of zero, i.e. silent corruption, not a
        // crash. This is the exact hazard the q4_k_ud bring-up was told to close
        // before widening the type surface this switch dispatches over.
        TORCH_CHECK(false, "ggml_moe_a8_vec: unsupported ggml type ", type);
    }
  });
  return Y;
}

// FreeToken: weight-reuse batched sibling of ``ggml_moe_a8_vec``.
//
// Identical semantics and -- deliberately -- identical output BITS. The only
// difference is scheduling: N routed rows that share an expert are handled by
// one CUDA block, so the expert's weight row is pulled from HBM once per N rows
// instead of once per row. See moe_vec_batched.cuh for the rationale and for why
// the batching cannot perturb a single float.
//
// It is exposed as a SEPARATE entry point rather than an extra argument on
// ``ggml_moe_a8_vec`` so the original kernel stays byte-for-byte reachable for
// A/B comparison (which is exactly what the equality test does).
//
// ``perm`` is an int32 CUDA tensor of routed-row indices grouped so that every
// aligned run of ``batch_n`` entries belongs to a single expert, with -1 in the
// alignment padding. Its length must be a multiple of ``batch_n`` and may be
// LONGER than ``tokens * top_k``: the host sizes it to a fixed worst-case bound
// (``routed + num_experts * (batch_n - 1)``, rounded up) so that building it
// never needs a device->host sync. The surplus entries are all -1 and their
// groups exit on the first instruction.
//
// ``out``, when supplied, is written in place instead of allocating a fresh
// zeroed tensor. Only real routed rows are ever stored to -- which is what lets
// a test poison the buffer and prove the padding lanes never wrote.
torch::Tensor ggml_moe_a8_vec_batched(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor topk_ids,
    torch::Tensor perm,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens,
    int64_t row_pitch_bytes,
    int64_t batch_n,
    std::optional<torch::Tensor> out) {
  TORCH_CHECK(
      moe_vec_batched_has_type(type),
      "ggml_moe_a8_vec_batched: no batched kernel for ggml type ",
      type,
      " (covered: 10 Q2_K, 17 IQ2_XS, 18 IQ3_XXS, 39 MXFP4). Use ggml_moe_a8_vec.");
  TORCH_CHECK(batch_n >= 2, "ggml_moe_a8_vec_batched: batch_n must be >= 2 (got ", batch_n, ")");
  TORCH_CHECK(
      perm.is_cuda() && perm.is_contiguous() && perm.scalar_type() == torch::kInt,
      "ggml_moe_a8_vec_batched: perm must be a contiguous int32 CUDA tensor");
  const int64_t padded_total = perm.numel();
  TORCH_CHECK(
      padded_total % batch_n == 0,
      "ggml_moe_a8_vec_batched: perm length (",
      padded_total,
      ") is not a multiple of batch_n (",
      batch_n,
      ")");

  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y;
  if (out.has_value()) {
    Y = out.value();
    TORCH_CHECK(
        Y.is_cuda() && Y.is_contiguous() && Y.scalar_type() == X.scalar_type() && Y.dim() == 2 &&
            Y.sizes()[0] == tokens * top_k && Y.sizes()[1] == row,
        "ggml_moe_a8_vec_batched: `out` must be a contiguous CUDA tensor of X's dtype "
        "with shape [tokens * top_k, row]");
  } else {
    Y = torch::zeros({tokens * top_k, row}, options);
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8_vec_batched", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 10:
        moe_vec_batched_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            (const int*)perm.data_ptr(),
            top_k,
            padded_total,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            batch_n,
            stream);
        break;
      case 17:
        moe_vec_batched_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            (const int*)perm.data_ptr(),
            top_k,
            padded_total,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            batch_n,
            stream);
        break;
      case 18:
        moe_vec_batched_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            (const int*)perm.data_ptr(),
            top_k,
            padded_total,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            batch_n,
            stream);
        break;
      case 39:
        moe_vec_batched_mxfp4_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            (const int*)perm.data_ptr(),
            top_k,
            padded_total,
            col,
            row,
            quant_X.stride(0),
            row_pitch_bytes,
            batch_n,
            stream);
        break;
      default:
        // Defense in depth: moe_vec_batched_has_type(type) above already rejects
        // any type without a case here, so this should be unreachable -- but an
        // unmatched case would otherwise silently return `out` (or a freshly
        // zeroed Y) untouched, the same silent-zero hazard ggml_moe_a8_vec has.
        TORCH_CHECK(false, "ggml_moe_a8_vec_batched: unsupported ggml type ", type);
    }
  });
  return Y;
}

// Which ggml types the batched path can serve. Callers check this and fall back
// to ``ggml_moe_a8_vec`` rather than eating a TORCH_CHECK on an exotic bank.
bool ggml_moe_vec_batched_supported(int64_t type) {
  return moe_vec_batched_has_type(type);
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}

// ``sizeof(block_q_t)`` for a ggml type -- the byte width of ONE quantized
// super-block, so ``blocks_per_row * ggml_type_block_bytes(t)`` is the natively
// packed row width.
//
// Deliberately NOT ``ggml_moe_get_block_size``: that one returns the MMQ tile
// height MOE_X_* (a row count), returns 0 for every IQ type, and 4 for Q2_K.
// Confusing the two silently truncates a bank slice, so they are separate
// functions with separate names.
//
// Used by the prefill dequant-GEMM path: the q2_k_ud banks pad every row out to
// one shared pitch, and slicing the native prefix off each row turns a padded
// bank back into the tightly-packed bank ``ggml_dequantize`` assumes.
int64_t ggml_type_block_bytes(int64_t type) {
  switch (type) {
    case 2:
      return (int64_t)sizeof(block_q4_0);
    case 3:
      return (int64_t)sizeof(block_q4_1);
    case 6:
      return (int64_t)sizeof(block_q5_0);
    case 7:
      return (int64_t)sizeof(block_q5_1);
    case 8:
      return (int64_t)sizeof(block_q8_0);
    case 10:
      return (int64_t)sizeof(block_q2_K);
    case 11:
      return (int64_t)sizeof(block_q3_K);
    case 12:
      return (int64_t)sizeof(block_q4_K);
    case 13:
      return (int64_t)sizeof(block_q5_K);
    case 14:
      return (int64_t)sizeof(block_q6_K);
    case 16:
      return (int64_t)sizeof(block_iq2_xxs);
    case 17:
      return (int64_t)sizeof(block_iq2_xs);
    case 18:
      return (int64_t)sizeof(block_iq3_xxs);
    case 19:
      return (int64_t)sizeof(block_iq1_s);
    case 20:
      return (int64_t)sizeof(block_iq4_nl);
    case 21:
      return (int64_t)sizeof(block_iq3_s);
    case 22:
      return (int64_t)sizeof(block_iq2_s);
    case 23:
      return (int64_t)sizeof(block_iq4_xs);
    case 29:
      return (int64_t)sizeof(block_iq1_m);
    case 39:
      return (int64_t)sizeof(block_mxfp4);  // 17
  }
  return 0;
}

// Elements per super-block (ggml's ``blck_size``) for a type. Pairs with
// ``ggml_type_block_bytes``: ``row_bytes = ncols / block_elems * block_bytes``.
// Kept here rather than in a Python table so the two halves of that product can
// never drift apart from the structs the dequant kernels actually read.
int64_t ggml_type_block_elems(int64_t type) {
  switch (type) {
    case 2:
      return QK4_0;
    case 3:
      return QK4_1;
    case 6:
      return QK5_0;
    case 7:
      return QK5_1;
    case 8:
      return QK8_0;
    case 20:
      return QK4_NL;
    case 39:
      return QK_MXFP4;
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 16:
    case 17:
    case 18:
    case 19:
    case 21:
    case 22:
    case 23:
    case 29:
      return QK_K;
  }
  return 0;
}

// ---- FreeToken pybind bindings (donor registers these via TORCH_LIBRARY; we
// expose them through torch.utils.cpp_extension.load's pybind module instead) ----
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "");
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "");
  m.def(
      "ggml_moe_a8",
      &ggml_moe_a8,
      "",
      py::arg("X"),
      py::arg("W"),
      py::arg("sorted_token_ids"),
      py::arg("expert_ids"),
      py::arg("num_tokens_post_padded"),
      py::arg("type"),
      py::arg("row"),
      py::arg("top_k"),
      py::arg("tokens"),
      py::arg("row_pitch_bytes") = 0);
  m.def(
      "ggml_moe_a8_vec",
      &ggml_moe_a8_vec,
      "",
      py::arg("X"),
      py::arg("W"),
      py::arg("topk_ids"),
      py::arg("top_k"),
      py::arg("type"),
      py::arg("row"),
      py::arg("tokens"),
      py::arg("row_pitch_bytes") = 0);
  m.def(
      "ggml_moe_a8_vec_batched",
      &ggml_moe_a8_vec_batched,
      "",
      py::arg("X"),
      py::arg("W"),
      py::arg("topk_ids"),
      py::arg("perm"),
      py::arg("top_k"),
      py::arg("type"),
      py::arg("row"),
      py::arg("tokens"),
      py::arg("row_pitch_bytes") = 0,
      py::arg("batch_n") = 8,
      py::arg("out") = py::none());
  m.def("ggml_moe_vec_batched_supported", &ggml_moe_vec_batched_supported, "");
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size, "");
  m.def("ggml_type_block_bytes", &ggml_type_block_bytes, "");
  m.def("ggml_type_block_elems", &ggml_type_block_elems, "");
}
