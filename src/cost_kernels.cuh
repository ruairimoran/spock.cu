#ifndef SPOCK_COST_KERNELS_CUH
#define SPOCK_COST_KERNELS_CUH

namespace cost_detail {

// Read each ancestor's coordinate directly into the cost output. Diagonal
// factors need neither a dense GEMM nor an intermediate gathered vector.
template<typename T>
__global__ void diagonalForward(T *output, const T *input, const T *factor,
                                const size_t *ancestors, size_t size, size_t nodes,
                                size_t stride, size_t offset, bool leaf) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t node = index / size + (leaf ? 0 : 1), coordinate = index % size;
    if (node >= nodes) return;
    const size_t source = leaf ? node : ancestors[node];
    output[node * stride + offset + coordinate] = factor[node * size + coordinate] * input[source * size + coordinate];
}

template<typename T>
__device__ T adjointEntry(const T *dual, const T *factor, const T *linear,
                          size_t node, size_t coordinate, size_t size,
                          size_t stride, size_t offset, size_t scalarOffset) {
    T value = factor[node * size + coordinate] * dual[node * stride + offset + coordinate];
    if (linear)
        value -= T(0.5) * linear[node * size + coordinate] *
            (dual[node * stride + scalarOffset] + dual[node * stride + scalarOffset + 1]);
    return value;
}

// One thread owns a destination coordinate and sums its children. This
// preserves accumulation into the primal vector without atomic updates.
template<typename T>
__global__ void diagonalAdjoint(T *output, const T *dual, const T *factor, const T *linear,
                                const size_t *childFrom, const size_t *numChildren,
                                size_t size, size_t destinations, size_t stride,
                                size_t offset, size_t scalarOffset, bool leaf) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size * destinations) return;
    const size_t parent = index / size, coordinate = index % size;
    T sum = 0;
    if (leaf) sum = adjointEntry(dual, factor, linear, parent, coordinate, size, stride, offset, scalarOffset);
    else
        for (size_t j = 0; j < numChildren[parent]; ++j)
            sum += adjointEntry(dual, factor, linear, childFrom[parent] + j, coordinate, size, stride, offset, scalarOffset);
    output[index] += sum;
}

template<typename T>
__global__ void linearAdjoint(T *output, const T *dual, const T *linear, size_t size,
                              size_t nodes, size_t stride, size_t scalarOffset, bool leaf) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size * nodes) return;
    const size_t node = index / size;
    if (!leaf && node == 0) return;
    output[index] -= T(0.5) * linear[index] *
        (dual[node * stride + scalarOffset] + dual[node * stride + scalarOffset + 1]);
}

template<typename T>
__global__ void forwardScalars(T *output, T *scalar, const T *x, const T *u,
                               const T *q, const T *r, const size_t *ancestors,
                               size_t nx, size_t nu, size_t nodes, size_t stride, bool leaf) {
    const size_t node = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    T value = T(0.5) * scalar[node];
    if (leaf || node > 0) {
        const size_t source = leaf ? node : ancestors[node];
        if (q)
            for (size_t i = 0; i < nx; ++i) value -= T(0.5) * q[node * nx + i] * x[source * nx + i];
        if (r)
            for (size_t i = 0; i < nu; ++i) value -= T(0.5) * r[node * nu + i] * u[source * nu + i];
        const size_t offset = leaf ? nx : nx + nu;
        output[node * stride + offset] = output[node * stride + offset + 1] = value;
    }
    scalar[node] = value;
}

template<typename T>
__global__ void adjointScalars(T *output, const T *dual, size_t nodes, size_t stride,
                               size_t offset, bool leaf) {
    const size_t node = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    if (leaf || node > 0)
        output[node] = T(0.5) * (dual[node * stride + offset] + dual[node * stride + offset + 1]);
    else output[node] *= T(0.5);
}

}  // namespace cost_detail
#endif
