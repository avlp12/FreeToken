#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <cstddef>
#include <cstdint>

// Host wrapper over cudaMemcpy2DAsync: copy ``height`` rows of ``width`` bytes from a
// pinned host buffer whose rows sit at ``spitch`` into a device buffer whose rows sit
// at ``dpitch``.
//
// WHY THIS AND NOT torch's OWN STRIDED COPY. A mixed-quant expert bank pads every
// weight row out to the widest ggml type in it, and the prefill double buffer only
// ever reads the native prefix. Skipping the tail is 24% of the gate_up bank's H2D.
// Expressed as ``dst[:, :, :w].copy_(src[:, :, :w])`` torch abandons the DMA engine
// and runs an elementwise kernel over pinned memory: measured 6.6 GB/s against the
// 50 GB/s of the contiguous copy, i.e. 5.5x SLOWER while moving fewer bytes. The copy
// engine's own 2D mode does it at 46.6 GB/s.
//
// One caveat worth stating because it is counter-intuitive and the caller depends on
// it: this is only worth using when ``width < spitch``. With width == pitch the same
// call collapses to ~7 GB/s on this driver -- a genuinely 2D descriptor for what is
// really one linear run -- so a caller with nothing to skip must stay on the plain
// contiguous copy. See OffloadMoeCache._prefill_layer_copy.
struct Memcpy2D {
    static void run(
        int64_t dst_ptr,
        int64_t dpitch,
        int64_t src_ptr,
        int64_t spitch,
        int64_t width,
        int64_t height,
        int64_t stream_handle
    ) {
        using namespace host;
        RuntimeCheck(width > 0 && height > 0, "memcpy2d: empty extent");
        RuntimeCheck(width <= dpitch && width <= spitch,
            "memcpy2d: width (", width, ") exceeds a pitch (", dpitch, ", ", spitch, ")");
        CUDA_CHECK(::cudaMemcpy2DAsync(
            reinterpret_cast<void*>(dst_ptr),
            static_cast<std::size_t>(dpitch),
            reinterpret_cast<const void*>(src_ptr),
            static_cast<std::size_t>(spitch),
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height),
            ::cudaMemcpyHostToDevice,
            reinterpret_cast<::cudaStream_t>(stream_handle)
        ));
    }
};
