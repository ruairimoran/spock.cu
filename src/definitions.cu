#include "../include/gpu.cuh"


/**
 * Definitions for CUDA kernels 
*/


/*
 * General
*/

__host__ __device__ size_t getIdxMat(size_t node, size_t row, size_t col, size_t rows, size_t cols = 0) {
    if (cols == 0) cols = rows;
    return (node * rows * cols) + (col * rows + row);
}

__global__ void d_setMatToId(real_t *mat, size_t numRows, size_t node = 0) {
    if (blockIdx.x == threadIdx.x) {
        size_t idx = getIdxMat(node, blockIdx.x, threadIdx.x, numRows);
        mat[idx] = 1.0;
    }
}


/*
 * Cache methods
*/


/*
 * Constraints methods
*/

__global__ void d_projectRectangle(size_t dimension, real_t *vec, real_t *lowerBound, real_t *upperBound) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dimension) {
        if (vec[i] < lowerBound[i]) vec[i] = lowerBound[i];
        if (vec[i] > upperBound[i]) vec[i] = upperBound[i];
    }
}


/*
 * Risk methods
*/


/*
 * Cone methods
*/

__global__ void d_maxWithZero(real_t *vec, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}

__global__ void d_setToZero(real_t *vec, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}

__global__ void d_projectOnSoc(real_t *vec, size_t n, real_t nrm, real_t scaling) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] *= scaling;
    if (i == n - 1) vec[i] = scaling * nrm;
}


/*
 * ScenarioTree methods
*/
