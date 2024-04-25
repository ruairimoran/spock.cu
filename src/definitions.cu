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

__global__ void d_avarVecAddB(real_t *vec, size_t node, size_t *numCh, size_t *chFrom, real_t *probs) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCh[node]) vec[i] += probs[chFrom[node] + i];
}


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

/**
 * Computing conditional probability of each tree node
 *
 * @param[in] anc device ptr to ancestor of node at index
 * @param[in] prob device ptr to probability of visiting node at index
 * @param[in] numNodes total number of nodes
 * @param[out] condProb device ptr to conditional probability of visiting node at index, given ancestor node visited
 */
__global__ void d_populateProbabilities(size_t *anc, real_t *prob, size_t numNodes, real_t *condProb) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        condProb[i] = 1.0;
    } else if (i < numNodes) {
        condProb[i] = prob[i] / prob[anc[i]];
    }
}

/**
 * Computing number of children of each tree node
 *
 * @param[in] from device ptr to first child of node at index
 * @param[in] to device ptr to last child of node at index
 * @param[in] numNonleafNodes total number of nonleaf nodes
 * @param[out] numChildren device ptr to number of children of node at index
 */
__global__ void d_populateChildren(size_t *from, size_t *to, size_t numNonleafNodes, size_t *numChildren) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonleafNodes) numChildren[i] = to[i] - from[i] + 1;
}

/**
 * Populating stagesFrom and stagesTo
 *
 * @param[in] stages device ptr to stage of node at index
 * @param[in] numStages total number of stages
 * @param[out] nodeFrom device ptr to first node of stage at index
 * @param[out] nodeTo device ptr to last node of stage at index
 */
__global__ void d_populateStages(size_t *stages, size_t numStages, size_t numNodes, size_t *nodeFrom, size_t *nodeTo) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numStages) {
        for (size_t j = 0; j < numNodes; j++) {
            if (stages[j] == i) {
                nodeFrom[i] = j;
                break;
            }
        }
        for (size_t j = numNodes - 1;; j--) {
            if (stages[j] == i) {
                nodeTo[i] = j;
                break;
            }
        }
    }
}
