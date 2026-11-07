#ifndef SPOCK_EXECUTION_CUH
#define SPOCK_EXECUTION_CUH

#include <tensor.cuh>
#include <stdexcept>

namespace spock {

enum class ProjectionMode { automatic, sequential, concurrent };

inline const char *projectionModeName(ProjectionMode mode) {
    switch (mode) {
        case ProjectionMode::automatic: return "automatic";
        case ProjectionMode::sequential: return "sequential";
        case ProjectionMode::concurrent: return "concurrent";
    }
    return "unknown";
}

// Stream index zero is GPUtils' first stream, not CUDA's legacy NULL stream.
// Custom kernels must use the same actual stream as their tensor operations.
template<typename T>
cudaStream_t executionStream(const DTensor<T> &tensor) {
    cudaStream_t stream;
    gpuErrChk(cublasGetStream(Session::getInstance().cuBlasHandle(tensor.streamIdx()), &stream));
    return stream;
}

template<typename T, typename... U>
cudaStream_t commonStream(const DTensor<T> &first, const DTensor<U> &...others) {
    const auto stream = executionStream(first);
    if (((executionStream(others) != stream) || ...))
        throw std::invalid_argument("SPOCK operands must select the same CUDA stream");
    return stream;
}

namespace detail {

// Host transfers remain completion boundaries: callers may immediately read
// the destination or reuse the source. Persistent GPU-to-GPU copies do not wait.
inline void copyToHost(void *host, const void *device, size_t bytes, cudaStream_t stream) {
    gpuErrChk(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream));
    gpuErrChk(cudaStreamSynchronize(stream));
}

inline void copyToDevice(void *device, const void *host, size_t bytes, cudaStream_t stream) {
    gpuErrChk(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream));
    gpuErrChk(cudaStreamSynchronize(stream));
}

inline void copy2DToHost(void *host, size_t hostPitch, const void *device, size_t devicePitch,
                         size_t width, size_t height, cudaStream_t stream) {
    gpuErrChk(cudaMemcpy2DAsync(host, hostPitch, device, devicePitch, width, height,
                              cudaMemcpyDeviceToHost, stream));
    gpuErrChk(cudaStreamSynchronize(stream));
}

}  // namespace detail
}  // namespace spock

#endif
