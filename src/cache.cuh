#ifndef __CACHE__
#define __CACHE__
#include "../include/stdgpu.h"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
class Cache {

private:
    ScenarioTree& m_tree;  ///< Previously created scenario tree of problem
    ProblemData& m_data;  ///< Previously created data of problem
    real_t m_tol = 0;
    size_t m_maxIters = 0;
    size_t m_countOperations = 0;

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree& tree, ProblemData& data, real_t tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        /** Transfer array data to device */
//        m_d_systemDynamics.upload(hostSystemDynamics);
    }

    /**
     * Destructor
     */
    ~Cache() {}

    /**
     * Getters
     */
    size_t maxIters() { return m_maxIters; }
    size_t& countOperations() { return m_countOperations; }

    /**
     * Debugging
     */
    void print(){
        std::cout << "hello from cache" << std::endl;

//        size_t len = m_numStates * m_numStates * m_tree.numNodes();
//        std::vector<real_t> hostData(len);
//        m_d_systemDynamics.download(hostData);
//        std::cout << "System dynamics (from device): ";
//        for (size_t i=0; i<len; i++) {
//            std::cout << hostData[i] << " ";
//        }

    }
};

#endif
