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
        int lower = (vec[i] < lowerBound[i]);
        int upper = (vec[i] > upperBound[i]);
        vec[i] = vec[i] * (1 - lower) * (1 - upper) + lower * lowerBound[i] + upper * upperBound[i];
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
__global__ void k_projectionMultiSocStep1(T *data,
                                          size_t numCones,
                                          size_t coneDimension,
                                          T *lastElementOfCones,
                                          T *squaredElements,
                                          T *norms,
                                          int *i2,
                                          int *i3,
                                          T *scaling) {
    /* Copy data to workspace */
    const unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int cone = blockIdx.y;
    const unsigned int allButLastElement = cone * coneDimension + thread;
    if (cone < numCones) {
        lastElementOfCones[cone] = data[coneDimension * (cone + 1) - 1];
    }
    if (cone < numCones && thread < coneDimension - 1) {
        T temp = data[allButLastElement];
        squaredElements[allButLastElement - cone] = temp * temp;
    }
    __syncthreads(); /* sync threads in each block */

    /* Since each block corresponds to a different SOC and the dimension of
     * each SOC is not going to be too large in general, the addition will
     * be performed by using atomicAdd. In order for the synchronisation to
     * be block-wise (and not device-wide), we will use shared memory.
     * For this reason we won't do any map-reduce-type summation.
     */
    extern __shared__ unsigned char mem[];
    T *sharedMem = reinterpret_cast<T *>(mem);
    if (cone < numCones && thread < coneDimension - 1) {
        sharedMem[thread] = squaredElements[cone * (coneDimension - 1) + thread];
    }
    __syncthreads();

    /* and now do the addition atomically */
    if (cone < numCones && thread < coneDimension - 1) {
        atomicAdd(&norms[cone], sharedMem[thread]);
    }
    __syncthreads();

    /* Final touch: apply the square root to determine the Euclidean norms */
    if (cone < numCones && thread == 0) {
        norms[cone] = sqrt(norms[cone]);
    }
    __syncthreads();

    /* populate sets i2 and i3 and compute scaling parameters */
    if (cone < numCones && thread == 0) {
        T nrm_j = norms[cone];
        T t_j = lastElementOfCones[cone];
        int i1 = nrm_j <= t_j;
        i2[cone] = nrm_j <= -t_j && !i1;
        i3[cone] = !i1 && !i2[cone];
        scaling[cone] = (nrm_j + t_j) / (2. * nrm_j + (1 - i3[cone]));
    }
}

template __global__ void
k_projectionMultiSocStep1(float *, size_t, size_t, float *, float *, float *, int *, int *, float *);

template __global__ void
k_projectionMultiSocStep1(double *, size_t, size_t, double *, double *, double *, int *, int *, double *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep2(T *data,
                                          size_t numCones,
                                          size_t coneDimension,
                                          int *i2) {
    const unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int cone = blockIdx.y;
    const unsigned int allButLastElement = cone * coneDimension + thread;
    if (cone < numCones && thread < coneDimension) data[allButLastElement] *= 1 - i2[cone];
}

template __global__ void k_projectionMultiSocStep2(float *, size_t, size_t, int *);

template __global__ void k_projectionMultiSocStep2(double *, size_t, size_t, int *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep3(T *data,
                                          size_t numCones,
                                          size_t coneDimension,
                                          T *norms,
                                          int *i2,
                                          int *i3,
                                          T *scaling) {
    const unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int cone = blockIdx.y;
    const unsigned int allButLastElement = cone * coneDimension + thread;
    if (cone < numCones && thread < coneDimension - 1) {
        T multiplier = i3[cone] * scaling[cone] + 1 - i3[cone];
        data[allButLastElement] *= multiplier;
    }
    if (cone < numCones && thread == coneDimension - 1) {
        int c_i2 = i2[cone];
        int c_i3 = i3[cone];
        int c_i1 = (1 - c_i2) * (1 - c_i3);
        data[allButLastElement] =
            c_i1 * data[allButLastElement] + (1 - c_i1) * (1 - c_i2) * (c_i3 * (scaling[cone] * norms[cone] - 1) + 1);
    }
}

template __global__ void k_projectionMultiSocStep3(float *, size_t, size_t, float *, int *, int *, float *);

template __global__ void k_projectionMultiSocStep3(double *, size_t, size_t, double *, int *, int *, double *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiIndexedNnoc(T *data, size_t n, int *idxNnoc, int *idx) {
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        idx[i] = idxNnoc[i] && data[i] < 0.;
        data[i] = (1 - idx[i]) * data[i];
    }
}

template __global__ void
k_projectionMultiIndexedNnoc(float *, size_t, int *, int *);

template __global__ void
k_projectionMultiIndexedNnoc(double *, size_t, int *, int *);
