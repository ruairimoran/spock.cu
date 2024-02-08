/** 
 * Definitions for CUDA kernels 
*/
#include "../include/stdgpu.h"


/** 
 * Cone methods
*/

__global__ void maxWithZero(real_t* vec, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}

__global__ void setToZero(real_t* vec, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}

__global__ void projectOnSocElse(real_t* vec, size_t n, real_t nrm, real_t last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] = last * (vec[i] / nrm);
    if (i == n - 1) vec[i] = last;
}

__global__ void projectOnSoc(real_t* vec, size_t n, real_t nrm) {
    if (nrm <= vec[n-1]) {
        // Do nothing!
    } else if (nrm <= -vec[n-1]) {
        setToZero<<<1, n>>>(vec, n);
    } else {
        real_t avg = (nrm + vec[n-1]) / 2.;
        projectOnSocElse<<<1, n>>>(vec, n, nrm, avg);
    }
}


/**
 * ScenarioTree methods
*/

/**
 * Computing conditional probability of each tree node
 * @param[in] anc device ptr to ancestor of node at index
 * @param[in] prob device ptr to probability of visiting node at index
 * @param[in] numNodes total number of nodes
 * @param[out] condProb device ptr to conditional probability of visiting node at index, given ancestor node visited
 */
static __global__ void populateProbabilities(size_t* anc, real_t* prob, size_t numNodes, real_t* condProb) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        condProb[i] = 1.0;
    } else if (i < numNodes) {
        condProb[i] = prob[i] / prob[anc[i]];
    }
}

/**
 * Computing number of children of each tree node
 * @param[in] from device ptr to first child of node at index
 * @param[in] to device ptr to last child of node at index
 * @param[in] numNonleafNodes total number of nonleaf nodes
 * @param[out] numChildren device ptr to number of children of node at index
 */
static __global__ void populateChildren(size_t* from, size_t* to, size_t numNonleafNodes, size_t* numChildren) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonleafNodes) numChildren[i] = to[i] - from[i] + 1;
}

/**
 * Populating stagesFrom and stagesTo
 * @param[in] stages device ptr to stage of node at index
 * @param[in] numStages total number of stages
 * @param[out] stageFrom device ptr to first node of stage at index
 * @param[out] stageTo device ptr to last node of stage at index
 */
static __global__ void populateStages(size_t* stages, size_t numStages, size_t numNodes, size_t* stageFrom, size_t* stageTo) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numStages) {
        for (size_t j=0; j<numNodes; j++) {
            if (stages[j] == i) {
                stageFrom[i] = j;
                break;
            }
        }
        for (size_t j=numNodes-1; ; j--) {
            if (stages[j] == i) {
                stageTo[i] = j;
                break;
            }
        }
    }
}


/** 
 * ProblemData methods
*/