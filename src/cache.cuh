#ifndef __CACHE__
#define __CACHE__
#include "../include/stdgpu.h"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


__global__ void d_vanillaCp(real_t tol, size_t maxIters, size_t* numIters);


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
    DeviceVector<real_t> m_d_prim;
    size_t sizeOfPrimal = 0;
    DeviceVector<real_t> m_d_dual;

    DeviceVector<size_t> m_d_countIterations;
    DeviceVector<real_t> m_d_cacheError;

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree& tree, ProblemData& data, real_t tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        sizeOfPrimal = m_tree.numNodes() * m_data.numStates()  ///< States of all nodes
                + m_tree.numNonleafNodes() * m_data.numInputs()  ///< Inputs of all nonleaf nodes
                + m_tree.numNonleafNodes() * m_tree.numEvents()  ///< Y for all nonleaf nodes
                + m_tree.numNodes()  ///< Tau for all child nodes
                + m_tree.numNodes();  ///< S for all child nodes

        /** Allocate memory on device */
        m_d_prim.allocateOnDevice(sizeOfPrimal);
        m_d_countIterations.allocateOnDevice(1);
        m_d_cacheError.allocateOnDevice(m_maxIters);

        /** Transfer array data to device */
//        m_d_tol.upload(vecTol);
    }

    /**
     * Destructor
     */
    ~Cache() {}

    /**
     * Getters
     */
    real_t tol() { return m_tol; }
    size_t maxIters() { return m_maxIters; }

    /**
     * Setters
     */
    void initialiseState(std::vector<real_t> initState) {
        if (initState.size() != m_data.numStates()) {
            std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                << " but given " << initState.size() << " states" << std::endl;
            throw std::invalid_argument("Incorrect dimension of initial state");
        }
        m_d_prim.upload(initState);
    }

    /**
     * Algorithms
     */
    void vanillaCp();

    /**
     * Debugging
     */
    void print();
};


void Cache::vanillaCp() {
    d_vanillaCp<<<1, 1>>>(m_tol, m_maxIters, m_d_countIterations.get());
}


void Cache::print() {
    std::cout << "Tolerance: " << m_tol << std::endl;
    std::cout << "Max iterations: " << m_maxIters << std::endl;
    std::vector<size_t> hostDataSize;
    std::vector<real_t> hostDataReal;

    hostDataSize.resize(1);
    m_d_countIterations.download(hostDataSize);
    std::cout << "Number iterations (from device): " << hostDataSize[0] << std::endl;

    hostDataReal.resize(sizeOfPrimal);
    m_d_prim.download(hostDataReal);
    std::cout << "Primal (from device): ";
    for (size_t i=0; i<sizeOfPrimal; i++) {
        std::cout << hostDataReal[i] << " ";
    }
    std::cout << std::endl;
}

#endif
