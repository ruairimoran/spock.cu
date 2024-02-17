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

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree& tree, ProblemData& data) : m_tree(tree), m_data(data) {
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
//    size_t numStates() { return m_numStates; }

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
