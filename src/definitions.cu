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

__global__ void projectOnSoc(real_t* vec, size_t n, real_t nrm, real_t scaling) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] *= scaling;
    if (i == n - 1) vec[i] = scaling * nrm;
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
__global__ void populateProbabilities(size_t* anc, real_t* prob, size_t numNodes, real_t* condProb) {
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
__global__ void populateChildren(size_t* from, size_t* to, size_t numNonleafNodes, size_t* numChildren) {
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
__global__ void populateStages(size_t* stages, size_t numStages, size_t numNodes, size_t* stageFrom, size_t* stageTo) {
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

/**
 * Populate risk at each nonleaf node
*/
__global__ void populateRisks(size_t numNonleaf, size_t* childFrom, size_t* childTo, char riskType, real_t alpha) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonleaf) {
        size_t chFrom = childFrom[i];
        size_t chTo = childTo[i];
        // eye = np.eye(self.__num_children);
        riskMatEi = np.vstack((alpha*eye, -eye, np.ones((1, num_children))));
        std::vector riskMatFi(2*chNum+1, 0.);

        std::vector<real_t> riskVecBi;
        for (size_t j=chFrom; j<=chTo; j++) riskVecBi.push_back(m_tree.conditionalProbabilities().fetchElementFromDevice(j));
        for (size_t j=chFrom; j<=chTo; j++) riskVecBi.push_back(0.);
        riskVecBi.push_back(0.);
    }
}
