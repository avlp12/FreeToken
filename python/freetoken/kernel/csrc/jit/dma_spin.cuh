#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <cstdint>
#include <tvm/ffi/container/tensor.h>

// --moe-copy-engine doorbell (see offload_cache.DmaCopyService): both the epoch-bump
// and the spin/wait kernels below are deliberately plain CUDA C++, NOT triton.
//
// The spin kernel must hold the compute stream in a device-side loop until a value
// written by a *different host thread's* CUDA stream becomes visible, which only
// works if every loop iteration genuinely re-reads device memory; triton's
// `tl.load(ptr, volatile=True)` inside a Python-level `while` only *conventionally*
// re-issues the load each iteration, with no hard guarantee against register-caching
// across iterations. A C++ `volatile` pointer dereference has no such ambiguity: the
// standard requires every access through a volatile lvalue to touch memory.
//
// More importantly (found while reproducing the actual hang, see
// /root/test_stage_wait_debug.py): triton kernel LAUNCHES themselves are not safe in
// this hot path either, independent of the spin semantics above. With the spin kernel
// already converted to C++, the doorbell protocol still deadlocked in plain EAGER
// mode (no CUDA graph capture involved) once enough back-to-back, un-synchronized
// doorbell rounds queued up that the DmaCopyService daemon thread had real backlog
// (its per-row copy loop, running on a second Python thread). A watchdog thread-stack
// dump caught the MAIN thread stuck inside
// `triton/compiler/compiler.py:_init_handles` (via `launch_metadata`), invoked from
// triton's `_dma_epoch_kernel` launch -- triton's lazy per-launch handle/module
// (re)initialization does not tolerate this concurrent, backlogged, multi-threaded
// CUDA usage pattern on this platform (RTX 5090 / WSL2 GPU-PV / WDDM). Replacing that
// one triton launch with a non-triton op made the identical repro pass cleanly;
// replacing it with this kernel (going through the same load_jit/tvm-ffi launch path
// already used elsewhere in this codebase for exactly this kind of hot, repeatedly
// launched kernel) removes triton from the doorbell path entirely.
//
// Round 2 (found while reproducing a real-hardware graph-REPLAY hang, see
// /root/test_dma_graph_replay.py): even with a GIL-free C++ daemon (dma_service.cuh),
// a CAPTURED graph chaining 2+ doorbell rounds together still deadlocks on its SECOND
// round, every time, regardless of total round count (4 through 48 all fail
// identically) -- so this was never about queue *depth*/backlog volume. The daemon's
// ack was correctly computed and its cudaMemcpyAsync issued (h_ack caught up to
// h_epoch), yet the compute stream never advanced past round 1's spin kernel. The only
// thing that changed between "ack computed" and "ack visible to the spin kernel" is
// that writing it required ANOTHER GPU command (a D2H->H2D relay through `d_done`,
// issued on the daemon's own stream) to actually execute on the device -- and that
// command was submitted, chronologically, *after* the entire multi-round graph's
// launch. On this platform's WDDM/GPU-PV submission model that is apparently enough
// for it to queue behind commands that logically cannot run yet (round 2+'s kernels,
// already enqueued by the single cudaGraphLaunch call but blocked on round 1), even
// though those commands live on a completely different stream -- a true circular wait
// at the driver/queue level, independent of the GIL or which thread issues the ack.
//
// The fix: the ack no longer goes through the command queue at all. `done_ptr` here is
// the GPU-visible alias of a *pinned host* tensor (computed once via
// freetoken.kernel.pinned.device_ptr, which is exactly the same host/device pointer
// translation the pre-existing zero-copy pull kernel already relies on for its bulk
// reads -- proven to work on this platform, just applied here to one 8-byte flag
// instead of multi-MB expert rows). The daemon acks with a bare host memory store
// (`h_ack[0] = e`; no cudaMemcpyAsync, no stream, no queue entry whatsoever); this
// kernel's `volatile` read of the SAME physical memory, from the GPU side, sees it via
// ordinary PCIe/NVLink host<->device cache coherency -- the same mechanism that makes
// zero-copy mapped memory work at all. Nothing about the epoch side needed to change:
// it is bumped by this stream's own dma_epoch_bump kernel and read back by this same
// kernel, a pure device-to-device dependency with no cross-queue relay involved.

namespace {

__global__ void dma_epoch_bump_kernel(int64_t* epoch_ptr, int32_t* layer_ptr, int32_t layer_id) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    *epoch_ptr += 1;
    *layer_ptr = layer_id;
}

__global__ void dma_spin_wait_kernel(int64_t done_addr, const int64_t* epoch_ptr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    // `done_addr` is a raw address (the mapped-host alias of the daemon's ack flag),
    // not a tvm-ffi tensor -- see the file comment above for why.
    auto* done = reinterpret_cast<const volatile int64_t*>(done_addr);
    auto* epoch = reinterpret_cast<const volatile int64_t*>(epoch_ptr);
    // Both sides of the comparison are re-read through `volatile` every iteration
    // (not just `done`): epoch is bumped once by dma_epoch_bump strictly before this
    // kernel is enqueued on the same stream, so a single snapshot would already be
    // correct -- but re-reading it too costs nothing and removes any dependence on
    // that stream-order assumption holding exactly as currently coded.
    int64_t d = *done;
    int64_t target = *epoch;
    while (d < target) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(1000);
#endif
        d = *done;
        target = *epoch;
    }
}

}  // namespace

inline void dma_epoch_bump_cpp(
    tvm::ffi::TensorView epoch, tvm::ffi::TensorView layer_out, int64_t layer_id
) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto epoch_dtype = SymbolicDType{};
    auto layer_dtype = SymbolicDType{};
    TensorMatcher({1}).with_dtype<int64_t>(epoch_dtype).with_device<kDLCUDA>(device).verify(epoch);
    TensorMatcher({1}).with_dtype<int32_t>(layer_dtype).with_device<kDLCUDA>(device)
        .verify(layer_out);
    auto* epoch_ptr = static_cast<int64_t*>(epoch.data_ptr());
    auto* layer_ptr = static_cast<int32_t*>(layer_out.data_ptr());
    LaunchKernel(1, 1, device.unwrap())(
        dma_epoch_bump_kernel, epoch_ptr, layer_ptr, static_cast<int32_t>(layer_id));
}

inline void dma_spin_wait_cpp(int64_t done_addr, tvm::ffi::TensorView epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto dtype = SymbolicDType{};
    TensorMatcher({1}).with_dtype<int64_t>(dtype).with_device<kDLCUDA>(device).verify(epoch);
    const auto* epoch_ptr = static_cast<const int64_t*>(epoch.data_ptr());
    LaunchKernel(1, 1, device.unwrap())(dma_spin_wait_kernel, done_addr, epoch_ptr);
}
