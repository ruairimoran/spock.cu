#ifndef SPOCK_DYNAMICS_KERNELS_CUH
#define SPOCK_DYNAMICS_KERNELS_CUH

#include "../include/gpu.cuh"

namespace spock::detail {

// The two factors stay in their existing column-major buffers. A tile covers
// 32 output rows and four independent column sums, providing multiple blocks
// per child even when a stage contains only a modest number of scenarios.
// The shared vector tile is reused by both output blocks; affine P c is added
// while loading it, without modifying q or allocating a stacked matrix.
template<typename T>
__global__ void backwardChildProducts(
    T *inputContributions, T *stateContributions,
    const T *bTranspose, const T *closedLoopTranspose,
    const T *q, const T *pc,
    size_t states, size_t controls, size_t childFrom,
    bool computeState) {
    constexpr unsigned int rows = 32;
    constexpr unsigned int columns = 4;
    constexpr unsigned int tileSize = rows * columns;
    __shared__ T shared[tileSize];
    const unsigned int lane = threadIdx.x;
    const unsigned int part = threadIdx.y;
    const unsigned int thread = part * rows + lane;
    const size_t child = childFrom + blockIdx.x;
    const size_t outputRow = size_t(blockIdx.y) * rows + lane;
    const size_t width = controls + (computeState ? states : 0);
    const bool valid = outputRow < width;
    const bool inputBlock = outputRow < controls;
    const size_t row = inputBlock ? outputRow : outputRow - controls;
    const size_t factorRows = inputBlock ? controls : states;
    const T *factor = inputBlock ? bTranspose : closedLoopTranspose;
    T sum = 0;
    for (size_t start = 0; start < states; start += tileSize) {
        const size_t column = start + thread;
        T value = 0;
        if (column < states) {
            value = q[child * states + column];
            if (pc) value += pc[child * states + column];
        }
        shared[thread] = value;
        __syncthreads();
        if (valid) {
            const size_t length = min(size_t(tileSize), states - start);
            for (size_t k = part; k < length; k += columns)
                sum += factor[(child * states + start + k) * factorRows + row] * shared[k];
        }
        __syncthreads();
    }
    shared[thread] = sum;
    __syncthreads();
    if (part == 0 && valid) {
        sum = ((shared[lane] + shared[rows + lane])
               + shared[2 * rows + lane]) + shared[3 * rows + lane];
        if (inputBlock) inputContributions[child * controls + row] = sum;
        else stateContributions[child * states + row] = sum;
    }
}

// Each thread owns one parent/output component. Summing children in their
// stored order avoids atomics and repeated writes to the parent workspace.
// The input block also forms the unsolved right-hand side u_bar - sum(B' y).
template<typename T>
__global__ void reduceBackwardContributions(
    T *rightHandSides, T *q, const T *inputs,
    const T *inputContributions, const T *stateContributions,
    const size_t *childFrom, const size_t *numChildren,
    size_t states, size_t controls, size_t parentFrom, size_t parentCount,
    bool computeState) {
    const size_t width = controls + (computeState ? states : 0);
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= parentCount * width) return;
    const size_t parent = parentFrom + index / width;
    const size_t component = index % width;
    const bool inputBlock = component < controls;
    const size_t row = inputBlock ? component : component - controls;
    const size_t stride = inputBlock ? controls : states;
    const T *source = inputBlock ? inputContributions : stateContributions;
    T sum = 0;
    for (size_t offset = 0; offset < numChildren[parent]; ++offset)
        sum += source[(childFrom[parent] + offset) * stride + row];
    if (inputBlock)
        rightHandSides[parent * controls + row] = inputs[parent * controls + row] - sum;
    else
        q[parent * states + row] = sum;
}

}  // namespace spock::detail
#endif
