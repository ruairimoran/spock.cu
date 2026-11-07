#ifndef SPOCK_PARTIAL_LIFTING_KERNELS_CUH
#define SPOCK_PARTIAL_LIFTING_KERNELS_CUH

#include "../include/gpu.cuh"

namespace spock::detail {

// F is stored as F' throughout. Gather one edge per chain so GPUtils can
// compose all chains of a coarse level in a single batched matrix product.
template<typename T>
__global__ void packChainFactors(T *packed, const T *fTranspose,
    const size_t *nodes, size_t states, size_t chains, size_t length, size_t position) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t matrixSize = states * states;
    if (index < chains * matrixSize) {
        const size_t chain = index / matrixSize;
        const size_t node = nodes[chain * (length + 1) + position];
        packed[index] = fTranspose[node * matrixSize + index % matrixSize];
    }
}

template<typename T>
__global__ void dynamicsInputOffsets(T *offsets, const T *ab, const T *d,
    const T *affine, const size_t *parents, size_t states, size_t controls, size_t nodes) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= (nodes - 1) * states) return;
    const size_t node = 1 + index / states, row = index % states;
    T sum = affine ? affine[node * states + row] : T(0);
    for (size_t col = 0; col < controls; ++col)
        sum += ab[(node * (states + controls) + states + col) * states + row]
               * d[parents[node] * controls + col];
    offsets[node * states + row] = sum;
}

// Batched GPUtils products feed this small child reduction. Computing each
// matrix-vector product per child avoids serialising a whole branch on one SM.
template<typename T>
__global__ void partialPrefixBackward(T *y, const T *sources, const T *contributions,
    const size_t *childFrom, const size_t *children, size_t states,
    size_t first, size_t count, size_t firstChild) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * states) return;
    const size_t parent = first + index / states, row = index % states;
    T result = sources[parent * states + row];
    for (size_t ch = 0; ch < children[parent]; ++ch)
        result += contributions[(childFrom[parent] + ch - firstChild) * states + row];
    y[parent * states + row] = result;
}

template<typename T>
__device__ T warpSum(T value) {
    for (int shift = 16; shift > 0; shift >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, shift);
    return value;
}

// One warp per output row makes reads of the stored F' contiguous when
// applying F. No cross-warp communication is needed in this kernel.
template<typename T>
__global__ void partialPrefixForward(T *statesOut, const T *offsets, const T *fTranspose,
    const size_t *parents, size_t states, size_t first, size_t count) {
    const size_t child = first + blockIdx.x;
    const size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (blockIdx.x >= count || row >= states) return;
    T sum = 0;
    for (size_t col = threadIdx.x; col < states; col += 32)
        sum += fTranspose[(child * states + row) * states + col]
               * statesOut[parents[child] * states + col];
    sum = warpSum(sum);
    if (threadIdx.x == 0) statesOut[child * states + row] = sum + offsets[child * states + row];
}

// All threads call this routine together; its caller synchronises before
// swapping the two shared vectors. A warp spans the input columns on the
// forward path; that path uses extra warps to reduce its per-warp row loop.
template<typename T, bool Forward>
__device__ void chainMatVec(const T *fTranspose, const T *value, T *result,
    const T *source, size_t states) {
    if constexpr (Forward) {
        for (size_t row = threadIdx.y; row < states; row += blockDim.y) {
            T sum = 0;
            for (size_t col = threadIdx.x; col < states; col += 32)
                sum += fTranspose[row * states + col] * value[col];
            sum = warpSum(sum);
            if (threadIdx.x == 0) result[row] = sum + source[row];
        }
    } else {
        const size_t thread = threadIdx.y * 32 + threadIdx.x;
        for (size_t row = thread; row < states; row += size_t(32) * blockDim.y) {
            T sum = 0;
            for (size_t col = 0; col < states; ++col)
                sum += fTranspose[col * states + row] * value[col];
            result[row] = sum + source[row];
        }
    }
}

// Each block owns a complete short chain segment. Interior reconstruction
// excludes both endpoints, whose values were solved on the coarse chain.
// This prevents adjacent segments from racing on their shared boundary.
template<typename T, bool Forward, bool Reconstruct>
__global__ void localChainRecurrence(T *values, T *summaries, const T *sources,
    const T *fTranspose, const size_t *nodes, size_t states, size_t chains,
    size_t length, size_t blockLength) {
    extern __shared__ __align__(16) unsigned char sharedStorage[];
    T *value = reinterpret_cast<T *>(sharedStorage), *next = value + states;
    const size_t thread = threadIdx.y * 32 + threadIdx.x;
    const size_t block = blockIdx.x, chain = block % chains, level = block / chains;
    const size_t start = level * blockLength;
    const int edges = int(min(blockLength, length - start));
    const size_t *path = nodes + chain * (length + 1) + start;
    const size_t initial = Forward ? path[Reconstruct ? 0 : 1]
                                   : path[Reconstruct ? edges : edges - 1];
    for (size_t row = thread; row < states; row += size_t(32) * blockDim.y)
        value[row] = Reconstruct ? values[initial * states + row] : sources[initial * states + row];
    __syncthreads();
    const int begin = Forward ? (Reconstruct ? 1 : 2) : (Reconstruct ? edges - 1 : edges - 2);
    const int end = Forward ? (Reconstruct ? edges - 1 : edges) : (Reconstruct ? 1 : 0);
    for (int pos = begin; Forward ? pos <= end : pos >= end; pos += Forward ? 1 : -1) {
        const size_t factorNode = path[Forward ? pos : pos + 1];
        chainMatVec<T, Forward>(fTranspose + factorNode * states * states,
            value, next, sources + path[pos] * states, states);
        __syncthreads();
        T *previous = value; value = next; next = previous;
        if constexpr (Reconstruct)
            for (size_t row = thread; row < states; row += size_t(32) * blockDim.y)
                values[path[pos] * states + row] = value[row];
        __syncthreads();
    }
    if constexpr (!Reconstruct)
        for (size_t row = thread; row < states; row += size_t(32) * blockDim.y)
            summaries[block * states + row] = value[row];
}

// Boundary propagation has no cross-chain dependencies. B=1 reads the
// original edge and source directly, providing a summary-free stage baseline.
template<typename T, bool Forward>
__global__ void chainBoundary(T *values, const T *sources, const T *summaries,
    const T *maps, const T *fTranspose, const size_t *nodes,
    size_t states, size_t chains, size_t length, size_t blockLength, size_t level) {
    const size_t chain = blockIdx.x, start = level * blockLength;
    const size_t end = min(start + blockLength, length);
    const size_t *path = nodes + chain * (length + 1);
    const size_t block = level * chains + chain;
    const size_t left = path[start], right = path[end];
    const T *matrix = blockLength == 1 ? fTranspose + right * states * states
                                      : maps + block * states * states;
    const size_t sourceNode = Forward ? right : left;
    const T *offset = blockLength == 1 ? sources + sourceNode * states : summaries + block * states;
    if constexpr (Forward) {
        const size_t row = size_t(blockIdx.y) * 4 + threadIdx.y;
        if (row >= states) return;
        T sum = 0;
        for (size_t col = threadIdx.x; col < states; col += 32)
            sum += matrix[row * states + col] * values[left * states + col];
        sum = warpSum(sum);
        if (threadIdx.x == 0) values[right * states + row] = offset[row] + sum;
    } else {
        const size_t row = size_t(blockIdx.y) * 128 + threadIdx.y * 32 + threadIdx.x;
        if (row >= states) return;
        T sum = 0;
        for (size_t col = 0; col < states; ++col)
            sum += matrix[col * states + row] * values[right * states + col];
        values[left * states + row] = offset[row] + sum;
    }
}

}  // namespace spock::detail
#endif
