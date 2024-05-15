#include "../include/gpu.cuh"


/**
 * Definitions for CUDA kernels.
 *
 * Instances of template functions (for all types needed)
 * must be created by 'explicit template instantiation'.
*/

__host__ __device__ size_t getIdxMat(size_t node, size_t row, size_t col, size_t rows, size_t cols = 0) {
    if (cols == 0) cols = rows;
    return (node * rows * cols) + (col * rows + row);
}

TEMPLATE_WITH_TYPE_T
__global__ void d_setMatToId(T *mat, size_t numRows, size_t node = 0) {
    if (blockIdx.x == threadIdx.x) {
        size_t idx = getIdxMat(node, blockIdx.x, threadIdx.x, numRows);
        mat[idx] = 1.0;
    }
}
template __global__ void d_setMatToId(float*, size_t, size_t);
template __global__ void d_setMatToId(double*, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void d_projectRectangle(size_t dimension, T *vec, T *lowerBound, T *upperBound) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dimension) {
        if (vec[i] < lowerBound[i]) vec[i] = lowerBound[i];
        if (vec[i] > upperBound[i]) vec[i] = upperBound[i];
    }
}
template __global__ void d_projectRectangle(size_t, float*, float*, float*);
template __global__ void d_projectRectangle(size_t, double*, double*, double*);

TEMPLATE_WITH_TYPE_T
__global__ void d_maxWithZero(T *vec, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}
template __global__ void d_maxWithZero(float*, size_t);
template __global__ void d_maxWithZero(double*, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void d_setToZero(T *vec, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}
template __global__ void d_setToZero(float*, size_t);
template __global__ void d_setToZero(double*, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void d_projectOnSoc(T *vec, size_t n, T nrm, T scaling) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] *= scaling;
    if (i == n - 1) vec[i] = scaling * nrm;
}
template __global__ void d_projectOnSoc(float*, size_t, float, float);
template __global__ void d_projectOnSoc(double*, size_t, double, double);
