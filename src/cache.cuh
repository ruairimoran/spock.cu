#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


__host__ __device__ size_t getIdxMat(size_t node, size_t row, size_t col, size_t rows, size_t cols = 0);


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
class Cache {

private:
    ScenarioTree &m_tree;  ///< Previously created scenario tree
    ProblemData &m_data;  ///< Previously created problem
    real_t m_tol = 0;
    size_t m_maxIters = 0;
    size_t m_countIterations = 0;
    size_t m_primSize;
    std::unique_ptr<DTensor<real_t>> m_d_prim = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_primPrev = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_x = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_u = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_y = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_t = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_s = nullptr;
    size_t m_dualSize = 0;
    std::unique_ptr<DTensor<real_t>> m_d_dual = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_dualPrev = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_cacheError = nullptr;
    /* Other */
    std::unique_ptr<DTensor<real_t>> m_d_q = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_d = nullptr;

    /**
     * Private methods
     */
    void projectOnDynamics();
    void projectOnKernel();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree &tree, ProblemData &data, real_t tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        /* Sizes */
        size_t sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
        size_t sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
        size_t sizeY = m_tree.numNonleafNodes() * m_tree.numEvents();  ///< Y for all nonleaf nodes
        size_t sizeT = m_tree.numNodes();  ///< T for all child nodes
        size_t sizeS = m_tree.numNodes();  ///< S for all child nodes
        m_primSize = sizeX + sizeU + sizeY + sizeT + sizeS;

        /* Allocate memory on device */
        m_d_prim = std::make_unique<DTensor<real_t>>(m_primSize, true);
        m_d_primPrev = std::make_unique<DTensor<real_t>>(m_primSize, true);
        m_d_dual = std::make_unique<DTensor<real_t>>(m_dualSize, true);
        m_d_dualPrev = std::make_unique<DTensor<real_t>>(m_dualSize, true);
        m_d_cacheError = std::make_unique<DTensor<real_t>>(m_maxIters, true);
        m_d_q = std::make_unique<DTensor<real_t>>(sizeX, true);
        m_d_d = std::make_unique<DTensor<real_t>>(sizeU, true);

        /* Slice primal */
        size_t rowAxis = 0;
        size_t start = 0;
        m_d_x = std::make_unique<DTensor<real_t>>(*m_d_prim, rowAxis, start, sizeX - 1);
        start += sizeX;
        m_d_u = std::make_unique<DTensor<real_t>>(*m_d_prim, rowAxis, start, start + sizeU - 1);
        start += sizeU;
        m_d_y = std::make_unique<DTensor<real_t>>(*m_d_prim, rowAxis, start, start + sizeY - 1);
        start += sizeY;
        m_d_t = std::make_unique<DTensor<real_t>>(*m_d_prim, rowAxis, start, start + sizeT - 1);
        start += sizeT;
        m_d_s = std::make_unique<DTensor<real_t>>(*m_d_prim, rowAxis, start, start + sizeS - 1);
    }

    ~Cache() {}

    /**
     * Public methods
     */
    void initialiseState(std::vector<real_t> initState);
    void vanillaCp(std::vector<real_t> *previousSolution=nullptr);

    /**
     * Debugging
     */
    void print();
};

void Cache::projectOnDynamics() {
    *m_d_x *= -1.;
    m_d_x->deviceCopyTo(*m_d_q);
}

void Cache::projectOnKernel() {

}

void Cache::initialiseState(std::vector<real_t> initState) {
    if (initState.size() != m_data.numStates()) {
        std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                  << " but given " << initState.size() << " states" << "\n";
        throw std::invalid_argument("Incorrect dimension of initial state");
    }
    DTensor<real_t> slicePrim(*m_d_prim, 0, 0, m_data.numStates() - 1);
    slicePrim.upload(initState);
}

void Cache::vanillaCp(std::vector<real_t> *previousSolution) {
    if (previousSolution) m_d_prim->upload(*previousSolution);
    for (size_t i = 0; i < m_maxIters; i++) {
        projectOnDynamics();
        projectOnKernel();
        /** update z_bar */
        /** update n_bar */
        /** update z */
        /** update n */
        /** compute error */
        /** check error */
        if ((*m_d_cacheError)(i) <= m_tol) {
            m_countIterations = i;
            break;
        }
    }
}

void Cache::print() {
    std::cout << "Tolerance: " << m_tol << "\n";
    std::cout << "Num iterations: " << m_countIterations << " of " << m_maxIters << "\n";
    std::cout << "Primal (from device): " << m_d_prim->tr();
}


#endif
