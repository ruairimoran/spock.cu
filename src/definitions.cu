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
__global__ void k_setMatToId(T *mat, size_t numRows, size_t node = 0) {
    if (blockIdx.x == threadIdx.x) {
        size_t idx = getIdxMat(node, blockIdx.x, threadIdx.x, numRows);
        mat[idx] = 1.0;
    }
}

template __global__ void k_setMatToId(float *, size_t, size_t);

template __global__ void k_setMatToId(double *, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t dimension, T *vec, T *lowerBound, T *upperBound) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dimension) {
        if (vec[i] < lowerBound[i]) vec[i] = lowerBound[i];
        if (vec[i] > upperBound[i]) vec[i] = upperBound[i];
    }
}

template __global__ void k_projectRectangle(size_t, float *, float *, float *);

template __global__ void k_projectRectangle(size_t, double *, double *, double *);

TEMPLATE_WITH_TYPE_T
__global__ void k_maxWithZero(T *vec, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}

template __global__ void k_maxWithZero(float *, size_t);

template __global__ void k_maxWithZero(double *, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_setToZero(T *vec, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}

template __global__ void k_setToZero(float *, size_t);

template __global__ void k_setToZero(double *, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectOnSoc(T *vec, size_t n, T nrm, T scaling) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] *= scaling;
    if (i == n - 1) vec[i] = scaling * nrm;
}

template __global__ void k_projectOnSoc(float *, size_t, float, float);

template __global__ void k_projectOnSoc(double *, size_t, double, double);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSoc_s1(T *data,
                                        size_t numCones,
                                        size_t coneDimension,
                                        T *t_ws,
                                        T *squaredElements_ws,
                                        T *norms,
                                        int *i2,
                                        int *i3,
                                        T *scaling) {
    /* Copy data to workspace */
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idSoc = blockIdx.y;
    const unsigned int s = idSoc * coneDimension + tid;
    if (idSoc < numCones) t_ws[idSoc] = data[coneDimension * (idSoc + 1) - 1];
    if (idSoc < numCones && tid < coneDimension - 1) {
        T temp = data[s];
        squaredElements_ws[s - idSoc] = temp * temp;
    }
    __syncthreads(); /* sync threads in each block */

    /* Since each block corresponds to a different SOC and the dimension of
     * each SOC is not going to be too large in general#, the addition will
     * be performed by using atomicAdd. In order for the synchronisation to
     * be block-wise (and not device-wide), we will use shared memory.
     * # for this reason we won't do any map-reduce-type summation
     */
    extern __shared__ __align__(sizeof(T)) unsigned char mem[];
    T *sharedMem = reinterpret_cast<T *>(mem);
    if (idSoc < numCones && tid < coneDimension - 1)
        sharedMem[tid] = squaredElements_ws[idSoc * (coneDimension - 1) + tid];
    __syncthreads();

    /* and now do the addition atomically */
    if (idSoc < numCones && tid < coneDimension - 1) {
        atomicAdd(&norms[idSoc], sharedMem[tid]);
    }
    __syncthreads();

    /* Final touch: apply the square root to determine the Euclidean norms */
    if (idSoc < numCones && tid == 0) {
        norms[idSoc] = sqrt(norms[idSoc]);
    }
    __syncthreads();

    /* populate sets i2 and i3 and compute scaling parameters */
    if (idSoc < numCones && tid == 0) {
        T nrm_j = norms[idSoc];
        T t_j = t_ws[idSoc];
        i2[idSoc] = nrm_j <= -t_j;
        i3[idSoc] = nrm_j > t_j && nrm_j > -t_j;  // should this not be < >
        scaling[idSoc] = (nrm_j + t_j) / (2 * nrm_j);
    }
}

template __global__ void
k_projectionMultiSoc_s1(float *, size_t, size_t, float *, float *, float *, int *, int *, float *);

template __global__ void
k_projectionMultiSoc_s1(double *, size_t, size_t, double *, double *, double *, int *, int *, double *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSoc_s2_i2(T *data,
                                           size_t numCones,
                                           size_t coneDimension,
                                           int *i2) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idSoc = blockIdx.y;
    const unsigned int s = idSoc * coneDimension + tid;
    if (idSoc < numCones && tid < coneDimension) data[s] *= 1 - i2[idSoc];
}

template __global__ void k_projectionMultiSoc_s2_i2(float *, size_t, size_t, int *);

template __global__ void k_projectionMultiSoc_s2_i2(double *, size_t, size_t, int *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSoc_s2_i3(T *data,
                                           size_t numCones,
                                           size_t coneDimension,
                                           T *norms,
                                           int *i2,
                                           int *i3,
                                           T *scaling) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idSoc = blockIdx.y;
    const unsigned int s = idSoc * coneDimension + tid;

    if (idSoc < numCones && tid < coneDimension - 1) {
        T multiplier = i3[idSoc] * scaling[idSoc] + 1 - i3[idSoc];
        data[s] *= multiplier;
    }
    if (idSoc < numCones && tid == coneDimension - 1) {
        int c_i2 = i2[idSoc];
        int c_i3 = i3[idSoc];
        int c_i1 = (1 - c_i2) * (1 - c_i3);
        data[s] = c_i1 * data[s] + (1 - c_i1) * (1 - c_i2) * (c_i3 * (scaling[idSoc] * norms[idSoc] - 1) + 1);
    }
}

template __global__ void k_projectionMultiSoc_s2_i3(float *, size_t, size_t, float *, int *, int *, float *);

template __global__ void k_projectionMultiSoc_s2_i3(double *, size_t, size_t, double *, int *, int *, double *);
