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
        const unsigned int idx = getIdxMat(node, blockIdx.x, threadIdx.x, numRows);
        mat[idx] = 1.0;
    }
}

template __global__ void k_setMatToId(float *, size_t, size_t);

template __global__ void k_setMatToId(double *, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t dimension, T *vec, T *lowerBound, T *upperBound) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}

template __global__ void k_maxWithZero(float *, size_t);

template __global__ void k_maxWithZero(double *, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_setToZero(T *vec, size_t n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}

template __global__ void k_setToZero(float *, size_t);

template __global__ void k_setToZero(double *, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_shiftDiagonal(T *dst, T *src, size_t rows, size_t cols = 0) {
    if (cols == 0) cols = rows;
    const unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int c = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int srcIdx = c * rows + r;
    const unsigned int dstIdx = (c + 1) * rows + (r + 1);
    if (r < rows - 1 && c < cols - 1) {
        dst[dstIdx] = src[srcIdx];
    }
}

template __global__ void k_shiftDiagonal(float *, float *, size_t, size_t);

template __global__ void k_shiftDiagonal(double *, double *, size_t, size_t);

/**
 * Copy node data to other in parallel.
 * Must be launched with <<<[>=nodeTo+1], [>=max(nodeSizeDst, nodeSizeSrc)]>>>.
 *
 * @param dst memory destination
 * @param src memory source
 * @param nodeFrom first node to copy data to
 * @param nodeTo last node to copy data to (inclusive)
 * @param numEl number of elements per node of data to copy
 * @param nodeSizeDst number of elements per node in destination data
 * @param nodeSizeSrc number of elements per node in source data
 * @param elFromDst first element in node data to copy to (0-indexed)
 * @param elFromSrc first element in node data to copy from (0-indexed)
 * @param ancestors array of ancestors of each node
 * @param chFrom array of first child of each node
 * @param chTo array of last child of each node
 */
TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyNode2Node(T *dst, T *src,
                                  size_t nodeFrom, size_t nodeTo, size_t numEl,
                                  size_t nodeSizeDst, size_t nodeSizeSrc,
                                  size_t elFromDst, size_t elFromSrc) {
    const unsigned int element = threadIdx.x;
    const unsigned int node = blockIdx.x;
    if (node >= nodeFrom && node <= nodeTo) {
        const unsigned int dstIdx = node * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = node * nodeSizeSrc + elFromSrc + element;
        if (element < numEl) dst[dstIdx] = src[srcIdx];
    }
}

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyAnc2Node(T *dst, T *src,
                                 size_t nodeFrom, size_t nodeTo, size_t numEl,
                                 size_t nodeSizeDst, size_t nodeSizeSrc,
                                 size_t elFromDst, size_t elFromSrc,
                                 size_t *ancestors) {
    const unsigned int element = threadIdx.x;
    const unsigned int node = blockIdx.x;
    if (node >= nodeFrom && node <= nodeTo) {
        const unsigned int dstIdx = node * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = ancestors[node] * nodeSizeSrc + elFromSrc + element;
        if (element < numEl) dst[dstIdx] = src[srcIdx];
    }
}

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyLeaf2ZeroLeaf(T *dst, T *src,
                                      size_t nodeFrom, size_t nodeTo, size_t numEl,
                                      size_t nodeSizeDst, size_t nodeSizeSrc,
                                      size_t elFromDst, size_t elFromSrc) {
    const unsigned int element = threadIdx.x;
    const unsigned int node = blockIdx.x;
    if (node >= nodeFrom && node <= nodeTo) {
        const unsigned int dstIdx = (node - nodeFrom) * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = node * nodeSizeSrc + elFromSrc + element;
        if (element < numEl) dst[dstIdx] = src[srcIdx];
    }
}

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyZeroLeaf2Leaf(T *dst, T *src,
                                      size_t nodeFrom, size_t nodeTo, size_t numEl,
                                      size_t nodeSizeDst, size_t nodeSizeSrc,
                                      size_t elFromDst, size_t elFromSrc) {
    const unsigned int element = threadIdx.x;
    const unsigned int node = blockIdx.x;
    if (node >= nodeFrom && node <= nodeTo) {
        const unsigned int dstIdx = node * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = (node - nodeFrom) * nodeSizeSrc + elFromSrc + element;
        if (element < numEl) dst[dstIdx] = src[srcIdx];
    }
}

template __global__ void
k_memCpyNode2Node(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyNode2Node(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyAnc2Node(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t *);

template __global__ void
k_memCpyAnc2Node(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t *);

template __global__ void
k_memCpyLeaf2ZeroLeaf(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyLeaf2ZeroLeaf(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyZeroLeaf2Leaf(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyZeroLeaf2Leaf(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

/**
 * Copies data from a child index.
 * Must be launched with <<<[>=nodeTo+1], [>=numEl]>>>.
 * Caution! Dst must not be src.
 *
 * @param dst destination data
 * @param src source data
 * @param nodeFrom first node to set
 * @param nodeTo last node to set
 * @param numEl number of elements in node data
 * @param chIdx index of child (0-indexed)
 * @param chFrom first child of each node
 * @param numCh number of children of each node
 * @param add choose to add data instead of copy (default=false)
 */
TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyCh2Node(T *dst, T *src,
                                size_t nodeFrom, size_t nodeTo, size_t numEl, size_t chIdx,
                                size_t *chFrom, size_t *numCh, bool add = false) {
    const unsigned int element = threadIdx.x;
    const unsigned int node = blockIdx.x;
    if (node >= nodeFrom && node <= nodeTo && chIdx < numCh[node]) {
        const unsigned int dstIdx = node * numEl + element;
        const unsigned int srcIdx = (chFrom[node] + chIdx) * numEl + element;
        if (element < numEl) {
            dst[dstIdx] = (add == true) ? dst[dstIdx] + src[srcIdx] : src[srcIdx];
        }
    }
}

template __global__ void
k_memCpyCh2Node(float *, float *, size_t, size_t, size_t, size_t, size_t *, size_t *, bool);

template __global__ void
k_memCpyCh2Node(double *, double *, size_t, size_t, size_t, size_t, size_t *, size_t *, bool);

/**
 * To project onto matrices' kernels, we need to gather
 * `vec[i] <- (y_i, t[ch(i)], s[ch(i)])` for all nonleaf nodes `i`.
 * The `y` data can be copied by `k_memCpyNode2Node`.
 * The `t` and `s` data can be copied by this kernel.
 * Must be launched with <<<numBlocks(numNodes, TPB), TPB>>>.
 *
 * @param dst vec tensor [m,1,k]
 * @param src `t` or `s` data
 * @param numNodes total number of nodes of tree
 * @param sizeDst size m of vec
 * @param elFrom first element in vec to start copying into
 * @param ancestors ancestor of each node
 * @param chFrom first child of each node
 */
TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyInTS(T *dst, T *src, size_t numNodes, size_t sizeDst, size_t elFrom,
                             size_t *ancestors, size_t *chFrom) {
    const unsigned int child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child > 0 && child < numNodes) {
        const unsigned int anc = ancestors[child];
        const unsigned int dstIdx = anc * sizeDst + elFrom + child - chFrom[anc];
        dst[dstIdx] = src[child];
    }
}

template __global__ void
k_memCpyInTS(float *, float *, size_t, size_t, size_t, size_t *, size_t *);

template __global__ void
k_memCpyInTS(double *, double *, size_t, size_t, size_t, size_t *, size_t *);

/**
 * After projection onto matrices' kernels, we need to disperse
 * `(y_i, t[ch(i)], s[ch(i)]) <- vec[i]` for all nonleaf nodes `i`.
 * The `y` data can be copied by `k_memCpyNode2Node`.
 * The `t` or `s` data can be copied by this kernel.
 * Must be launched with <<<numBlocks(numNodes, TPB), TPB>>>.
 *
 * @param dst `t` or `s` data
 * @param src vec tensor [m,1,k]
 * @param numNodes total number of nodes of tree
 * @param sizeDst size m of vec
 * @param elFrom first element in vec to start copying from
 * @param ancestors ancestor of each node
 * @param chFrom first child of each node
 */
TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyOutTS(T *dst, T *src, size_t numNodes, size_t sizeSrc, size_t elFrom,
                              size_t *ancestors, size_t *chFrom) {
    const unsigned int child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child > 0 && child < numNodes) {
        const unsigned int anc = ancestors[child];
        const unsigned int srcIdx = anc * sizeSrc + elFrom + child - chFrom[anc];
        dst[child] = src[srcIdx];
    }
}

template __global__ void
k_memCpyOutTS(float *, float *, size_t, size_t, size_t, size_t *, size_t *);

template __global__ void
k_memCpyOutTS(double *, double *, size_t, size_t, size_t, size_t *, size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectOnSoc(T *vec, size_t n, T nrm, T scaling) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
