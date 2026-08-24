#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <tvm/ffi/container/tensor.h>

// --moe-copy-engine host daemon, round 2: a GIL-independent replacement for the
// Python `DmaCopyService._serve` thread.
//
// Round 1 fixed the eager-mode boot hang (a WDDM/GPU-PV submission-queue deadlock
// triggered by many un-synchronized doorbell rounds) by draining every eager round
// before returning. That throttle is a no-op once graphs are captured -- a captured
// graph replays as ONE atomic driver submission with no Python in between rounds, so
// there is nothing to synchronize against between them. Real hardware showed replay
// of a 43+-round graph still deadlocks: `cudaGraphLaunch` itself can block inside the
// driver (same WDDM/GPU-PV submission-queue pressure, now from a single graph
// carrying dozens of doorbell rounds instead of many separate eager launches), and
// that blocked call is not documented/guaranteed to release the GIL the way a
// blocking *synchronize* call is (graph launch is normally-async, so PyTorch's binding
// has no reason to wrap it in a GIL-release section). If it doesn't release the GIL,
// the Python `_serve` thread can never run again -- not even to read a pinned-memory
// int -- producing exactly the observed circular wait: spin kernel waits on the ack;
// the ack needs the Python daemon; the Python daemon needs the GIL; the GIL is held by
// the blocked graph-launch call; the graph-launch call is blocked behind the spin
// kernel it's trying to get past.
//
// The fix here removes the GIL from that cycle entirely: the daemon is a plain
// std::thread, started once from Python at DmaCopyService construction time and never
// touching a Python/CPython object again. It polls the pinned epoch doorbell (volatile
// host reads) and issues cudaMemcpyAsync per (bank, row) on its own dedicated,
// non-blocking stream (created here, not through torch). Errors are surfaced via a
// plain pinned int64 flag (`h_error`) that stage_and_wait already polls the same way
// it polls everything else pinned -- no callback into Python from the daemon thread is
// needed or attempted.
//
// The GIL fix alone was NOT sufficient, though (see dma_spin.cuh's file comment for
// the full story): even a GIL-free daemon's ack, if delivered via a cudaMemcpyAsync
// command on its own stream, can get stuck behind an entire multi-round graph's
// command stream on this platform's WDDM/GPU-PV submission model -- a real hardware/
// driver-level queue-ordering deadlock, not a host-scheduling one. So the ack here is
// a bare pinned-host memory store (`*h_ack_ptr = e`), never a CUDA API call at all --
// there is nothing for it to queue behind. The spin kernel reads the SAME physical
// memory through its GPU-visible mapped alias (see dma_spin.cuh).

namespace {

struct DmaServiceState {
    std::thread thread;
    std::atomic<bool> stop_flag{false};

    // Per-bank descriptors, copied out of the init-time CPU tensors so the thread
    // never has to touch a torch::Tensor / tvm::ffi object again.
    std::vector<int64_t> host_ptrs;  // [num_banks * num_layers], row-major by bank
    std::vector<int64_t> row_bytes;  // [num_banks]
    std::vector<int64_t> slot_ptrs;  // [num_banks], device base addr of each slot cache
    int64_t num_layers = 0;
    int64_t num_banks = 0;
    int64_t max_rows = 0;
    int64_t device_id = 0;

    // Raw pointers into pinned host / device memory owned by the Python
    // DmaCopyService for its whole lifetime -- valid for as long as this state is.
    int64_t* h_epoch_ptr = nullptr;
    int32_t* h_layer_ptr = nullptr;
    int64_t* h_num_ptr = nullptr;
    int32_t* h_slots_ptr = nullptr;
    int32_t* h_rows_ptr = nullptr;
    int64_t* h_ack_ptr = nullptr;   // pinned host memory; the spin kernel reads its
                                    // GPU-visible mapped alias directly (see
                                    // dma_spin.cuh) -- acking is a plain host store,
                                    // never a CUDA API call.
    int64_t* h_error_ptr = nullptr;
};

void dma_service_run(DmaServiceState* state) {
    if (::cudaSetDevice(static_cast<int>(state->device_id)) != cudaSuccess) {
        *state->h_error_ptr = -2;
        return;
    }
    cudaStream_t stream = nullptr;
    if (::cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        *state->h_error_ptr = -3;
        return;
    }

    auto* epoch = reinterpret_cast<volatile int64_t*>(state->h_epoch_ptr);
    int64_t last = 0;

    // Best-effort: unblock whichever spin kernel is currently waiting on epoch `e` so
    // the caller observes progress (and raises on its NEXT stage_and_wait call, per
    // h_error) instead of the compute stream spinning forever. Reachable from every
    // failure path below, including protocol corruption (invalid n/L), where nothing
    // was ever queued to ack it otherwise. A bare host store -- not a CUDA API call --
    // so there is no queue entry for it to get stuck behind even if the device/stream
    // state is already unusable (the whole reason h_ack is acked this way and not via
    // cudaMemcpyAsync, see the file comment above).
    auto recovery_ack = [&](int64_t e) { *state->h_ack_ptr = e; };

    while (!state->stop_flag.load(std::memory_order_relaxed)) {
        int64_t e = *epoch;
        if (e == last) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        last = e;

        int64_t n = *reinterpret_cast<volatile int64_t*>(state->h_num_ptr);
        int32_t L = *reinterpret_cast<volatile int32_t*>(state->h_layer_ptr);
        cudaError_t rc = cudaSuccess;

        if (n < 0 || n > state->max_rows || L < 0 || L >= state->num_layers) {
            // Protocol corruption (should be unreachable) -- record, unblock whatever
            // is spinning on this epoch, and stop rather than issuing copies against
            // out-of-range indices.
            *state->h_error_ptr = -1;
            recovery_ack(e);
            break;
        }

        for (int64_t b = 0; b < state->num_banks && rc == cudaSuccess; ++b) {
            const char* bank_host_base =
                reinterpret_cast<const char*>(state->host_ptrs[b * state->num_layers + L]);
            char* bank_slot_base = reinterpret_cast<char*>(state->slot_ptrs[b]);
            int64_t bytes = state->row_bytes[b];
            for (int64_t i = 0; i < n; ++i) {
                int32_t slot = state->h_slots_ptr[i];
                int32_t row = state->h_rows_ptr[i];
                rc = ::cudaMemcpyAsync(
                    bank_slot_base + static_cast<size_t>(slot) * bytes,
                    bank_host_base + static_cast<size_t>(row) * bytes,
                    static_cast<size_t>(bytes), cudaMemcpyHostToDevice, stream);
                if (rc != cudaSuccess) {
                    break;
                }
            }
        }

        if (rc == cudaSuccess) {
            // The ack is now a bare host store (see the file comment: no CUDA API call
            // can be allowed to sit in a queue behind the graph), which loses the
            // stream-order guarantee the old "ack via cudaMemcpyAsync on this same
            // stream" design got for free -- a store to pinned host memory is NOT
            // ordered against this stream's still-in-flight row copies at all. Make
            // that ordering explicit instead: block until every row copy above has
            // actually landed on the device before the ack becomes visible, so the
            // spin kernel releasing the compute stream always implies the rows are
            // really there. This only waits on the daemon's OWN small, dedicated
            // stream (a handful of row copies, not the compute stream's giant
            // multi-round graph), so it is not exposed to the same submission-queue
            // pressure this whole redesign exists to avoid.
            rc = ::cudaStreamSynchronize(stream);
        }
        if (rc == cudaSuccess) {
            *state->h_ack_ptr = e;
        }

        if (rc != cudaSuccess) {
            *state->h_error_ptr = static_cast<int64_t>(rc);
            recovery_ack(e);
            break;
        }
    }

    ::cudaStreamDestroy(stream);
}

}  // namespace

inline int64_t dma_service_start(
    tvm::ffi::TensorView host_ptrs,
    tvm::ffi::TensorView row_bytes,
    tvm::ffi::TensorView slot_ptrs,
    tvm::ffi::TensorView h_epoch,
    tvm::ffi::TensorView h_layer,
    tvm::ffi::TensorView h_num,
    tvm::ffi::TensorView h_slots,
    tvm::ffi::TensorView h_rows,
    tvm::ffi::TensorView h_ack,
    tvm::ffi::TensorView h_error,
    int64_t num_layers,
    int64_t num_banks,
    int64_t max_rows,
    int64_t device_id
) {
    using namespace host;
    RuntimeCheck(host_ptrs.numel() == num_banks * num_layers,
                 "dma_service_start: host_ptrs size mismatch");
    RuntimeCheck(row_bytes.numel() == num_banks, "dma_service_start: row_bytes size mismatch");
    RuntimeCheck(slot_ptrs.numel() == num_banks, "dma_service_start: slot_ptrs size mismatch");
    RuntimeCheck(h_slots.numel() >= max_rows, "dma_service_start: h_slots too small");
    RuntimeCheck(h_rows.numel() >= max_rows, "dma_service_start: h_rows too small");

    auto* state = new DmaServiceState();
    state->num_layers = num_layers;
    state->num_banks = num_banks;
    state->max_rows = max_rows;
    state->device_id = device_id;

    const int64_t* host_ptrs_data = static_cast<const int64_t*>(host_ptrs.data_ptr());
    state->host_ptrs.assign(host_ptrs_data, host_ptrs_data + num_banks * num_layers);
    const int64_t* row_bytes_data = static_cast<const int64_t*>(row_bytes.data_ptr());
    state->row_bytes.assign(row_bytes_data, row_bytes_data + num_banks);
    const int64_t* slot_ptrs_data = static_cast<const int64_t*>(slot_ptrs.data_ptr());
    state->slot_ptrs.assign(slot_ptrs_data, slot_ptrs_data + num_banks);

    state->h_epoch_ptr = static_cast<int64_t*>(h_epoch.data_ptr());
    state->h_layer_ptr = static_cast<int32_t*>(h_layer.data_ptr());
    state->h_num_ptr = static_cast<int64_t*>(h_num.data_ptr());
    state->h_slots_ptr = static_cast<int32_t*>(h_slots.data_ptr());
    state->h_rows_ptr = static_cast<int32_t*>(h_rows.data_ptr());
    state->h_ack_ptr = static_cast<int64_t*>(h_ack.data_ptr());
    state->h_error_ptr = static_cast<int64_t*>(h_error.data_ptr());

    *state->h_error_ptr = 0;
    state->thread = std::thread(dma_service_run, state);
    return reinterpret_cast<int64_t>(state);
}

inline void dma_service_stop(int64_t handle) {
    if (handle == 0) {
        return;
    }
    auto* state = reinterpret_cast<DmaServiceState*>(handle);
    state->stop_flag.store(true, std::memory_order_relaxed);
    if (state->thread.joinable()) {
        state->thread.join();
    }
    delete state;
}
