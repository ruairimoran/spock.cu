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

/**
 * Project vector in place on rectangle bounds of the form:
 * lb <= v <= ub
 * Launched with <<<numBlocks(n, TPB), TPB>>>.
 *
 * @param n size of vec
 * @param v vector
 * @param lb lower bound
 * @param ub upper bound
 */
TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t n, T *v, T *lb, T *ub) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int lower = (v[i] <= lb[i]);
        int upper = (v[i] >= ub[i]);
        T lbi = isinf(lb[i]) ? 0 : lower * lb[i];
        T ubi = isinf(ub[i]) ? 0 : upper * ub[i];
        v[i] = v[i] * (1 - lower) * (1 - upper) + lbi + ubi;
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
 * Caution! Max block size is 32x32=1024 threads per block - do not exceed!
 * Caution! Must be launched with <<<dim3(numBlocks(nodeTo-nodeFrom+1, t), numBlocks(numEl, t)), dim3(t, t)>>>,
 * where t<=32.
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
 */
TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyNode2Node(T *dst, T *src,
                                  size_t nodeFrom, size_t nodeTo, size_t numEl,
                                  size_t nodeSizeDst, size_t nodeSizeSrc,
                                  size_t elFromDst, size_t elFromSrc) {
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x + nodeFrom;
    const unsigned int element = blockIdx.y * blockDim.y + threadIdx.y;
    if (node >= nodeFrom && node <= nodeTo && element < numEl) {
        const unsigned int dstIdx = node * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = node * nodeSizeSrc + elFromSrc + element;
        dst[dstIdx] = src[srcIdx];
    }
}

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyAnc2Node(T *dst, T *src,
                                 size_t nodeFrom, size_t nodeTo, size_t numEl,
                                 size_t nodeSizeDst, size_t nodeSizeSrc,
                                 size_t elFromDst, size_t elFromSrc,
                                 const size_t *ancestors) {
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x + nodeFrom;
    const unsigned int element = blockIdx.y * blockDim.y + threadIdx.y;
    if (node >= nodeFrom && node <= nodeTo && element < numEl) {
        const unsigned int dstIdx = node * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = ancestors[node] * nodeSizeSrc + elFromSrc + element;
        dst[dstIdx] = src[srcIdx];
    }
}

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyLeaf2Zero(T *dst, T *src,
                                  size_t nodeFrom, size_t nodeTo, size_t numEl,
                                  size_t nodeSizeDst, size_t nodeSizeSrc,
                                  size_t elFromDst, size_t elFromSrc) {
    const unsigned int zeroIdxNode = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int node = zeroIdxNode + nodeFrom;
    const unsigned int element = blockIdx.y * blockDim.y + threadIdx.y;
    if (node >= nodeFrom && node <= nodeTo && element < numEl) {
        const unsigned int dstIdx = zeroIdxNode * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = node * nodeSizeSrc + elFromSrc + element;
        dst[dstIdx] = src[srcIdx];
    }
}

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyZero2Leaf(T *dst, T *src,
                                  size_t nodeFrom, size_t nodeTo, size_t numEl,
                                  size_t nodeSizeDst, size_t nodeSizeSrc,
                                  size_t elFromDst, size_t elFromSrc) {
    const unsigned int zeroIdxNode = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int node = zeroIdxNode + nodeFrom;
    const unsigned int element = blockIdx.y * blockDim.y + threadIdx.y;
    if (node >= nodeFrom && node <= nodeTo && element < numEl) {
        const unsigned int dstIdx = node * nodeSizeDst + elFromDst + element;
        const unsigned int srcIdx = zeroIdxNode * nodeSizeSrc + elFromSrc + element;
        dst[dstIdx] = src[srcIdx];
    }
}

template __global__ void
k_memCpyNode2Node(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyNode2Node(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyAnc2Node(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t *);

template __global__ void
k_memCpyAnc2Node(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t *);

template __global__ void
k_memCpyLeaf2Zero(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyLeaf2Zero(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyZero2Leaf(float *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template __global__ void
k_memCpyZero2Leaf(double *, double *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

/**
 * Copies data from a child index.
 * Caution! Dst must not be src.
 * Caution! Max block size is 32x32=1024 threads per block - do not exceed!
 * Caution! Must be launched with <<<dim3(numBlocks(nodeTo-nodeFrom+1, t), numBlocks(numEl, t)), dim3(t, t)>>>,
 * where t<=32.
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
                                const size_t *chFrom, const size_t *numCh, bool add = false) {
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x + nodeFrom;
    const unsigned int element = blockIdx.y * blockDim.y + threadIdx.y;
    if (node >= nodeFrom && node <= nodeTo && element < numEl && chIdx < numCh[node]) {
        const unsigned int dstIdx = node * numEl + element;
        const unsigned int srcIdx = (chFrom[node] + chIdx) * numEl + element;
        dst[dstIdx] = add ? dst[dstIdx] + src[srcIdx] : src[srcIdx];
    }
}

template __global__ void
k_memCpyCh2Node(float *, float *, size_t, size_t, size_t, size_t, const size_t *, const size_t *, bool);

template __global__ void
k_memCpyCh2Node(double *, double *, size_t, size_t, size_t, size_t, const size_t *, const size_t *, bool);

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
__global__ void k_projectionMultiSocStep0(T *data,
                                          size_t numCones,
                                          size_t coneDimension,
                                          T *lastElementOfCones,
                                          T *norms) {
    /* Copy data to workspace */
    const unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int cone = blockIdx.y;
    const unsigned int element = cone * coneDimension + thread;
    if (cone < numCones && thread == 0) {
        lastElementOfCones[cone] = data[coneDimension * (cone + 1) - 1];
    }

    /* Since each column of blocks (in the x-dimension)
     * corresponds to a different SOC and the number of
     * these blocks is not going to be too large in general,
     * the addition will be performed using block-wise atomicAdd.
     * Then, we add the few (if any) block results for
     * each cone (no need for map-reduce-type summation).
     * For faster block-wise addition, we will use shared memory.
     * CAUTION! Shared memory is 0-indexed PER BLOCK!
     */
    extern __shared__ char sharedArr[];
    T *sharedMem = reinterpret_cast<T *>(&sharedArr);
    if (cone < numCones && thread < coneDimension - 1) {
        T temp = data[element];
        sharedMem[threadIdx.x] = temp * temp;
    }
    __syncthreads();  // Sync threads in each block

    /* Do the block-wise addition atomically */
    if (cone < numCones && thread < coneDimension - 1) {
        atomicAdd(&norms[cone + blockIdx.x * numCones], sharedMem[threadIdx.x]);
    }

    /*
     * We break here as grid-wise synchronisation is required.
     */
}

template __global__ void
k_projectionMultiSocStep0(float *, size_t, size_t, float *, float *);

template __global__ void
k_projectionMultiSocStep0(double *, size_t, size_t, double *, double *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep1(size_t numCones,
                                          size_t blocksPerCone,
                                          T *lastElementOfCones,
                                          T *norms,
                                          int *i2,
                                          int *i3,
                                          T *scaling) {
    const unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int cone = blockIdx.y;
    /* Add the results of each block for each cone */
    if (cone < numCones && thread > 0 && thread < blocksPerCone) {
        atomicAdd(&norms[cone], norms[cone + thread * numCones]);
    }
    __syncthreads();

    /* Apply the square root to determine the Euclidean norms */
    if (cone < numCones && thread == 0) {
        norms[cone] = sqrt(norms[cone]);
    }
    __syncthreads();

    /* Populate sets i2 and i3 and compute scaling parameters */
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
k_projectionMultiSocStep1(size_t, size_t, float *, float *, int *, int *, float *);

template __global__ void
k_projectionMultiSocStep1(size_t, size_t, double *, double *, int *, int *, double *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep2(T *data,
                                          size_t numCones,
                                          size_t coneDimension,
                                          int *i2) {
    const unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int cone = blockIdx.y;
    const unsigned int element = cone * coneDimension + thread;
    /* If i2 is true, set all elements of cone to 0 */
    if (cone < numCones && thread < coneDimension) data[element] *= 1 - i2[cone];
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
    const unsigned int element = cone * coneDimension + thread;
    if (cone < numCones && thread < coneDimension - 1) {
        T multiplier = i3[cone] * scaling[cone] + 1 - i3[cone];
        data[element] *= multiplier;
    }
    if (cone < numCones && thread == coneDimension - 1) {
        int c_i2 = i2[cone];
        int c_i3 = i3[cone];
        int c_i1 = (1 - c_i2) * (1 - c_i3);
        data[element] = c_i1 * data[element] + (1 - c_i1) * (1 - c_i2) * (c_i3 * (scaling[cone] * norms[cone] - 1) + 1);
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
