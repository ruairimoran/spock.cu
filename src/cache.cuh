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
    size_t m_matAxis = 2;
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
    /* Host data */
    std::unique_ptr<std::vector<size_t>> m_childFrom = nullptr;  ///< Ptr to first node of stage at index
    std::unique_ptr<std::vector<size_t>> m_childTo = nullptr;  ///< Ptr to last node of stage at index
    std::unique_ptr<std::vector<size_t>> m_nodeFrom = nullptr;  ///< Ptr to first node of stage at index
    std::unique_ptr<std::vector<size_t>> m_nodeTo = nullptr;  ///< Ptr to last node of stage at index

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
        m_d_q = std::make_unique<DTensor<real_t>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<real_t>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);

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

        /* Host data */
        m_childFrom = std::make_unique<std::vector<size_t>>(m_tree.childFrom().numEl());
        m_tree.childFrom().download(*m_childFrom);
        m_childTo = std::make_unique<std::vector<size_t>>(m_tree.childTo().numEl());
        m_tree.childTo().download(*m_childTo);
        m_nodeFrom = std::make_unique<std::vector<size_t>>(m_tree.nodeFrom().numEl());
        m_tree.nodeFrom().download(*m_nodeFrom);
        m_nodeTo = std::make_unique<std::vector<size_t>>(m_tree.nodeTo().numEl());
        m_tree.nodeTo().download(*m_nodeTo);
    }

    ~Cache() {}

    size_t solutionSize() { return m_primSize; }

    /**
     * Public methods
     */
    void cpIter();
    void vanillaCp(std::vector<real_t> initState, std::vector<real_t> *previousSolution=nullptr);

    /**
     * Debugging
     */
    void print();
};

void Cache::projectOnDynamics() {
    *m_d_x *= -1.;
    m_d_x->deviceCopyTo(*m_d_q);
    for (size_t stage=m_tree.numStages()-1; stage>0; stage--) {
        size_t chNodeFr = (*m_nodeFrom)[stage];
        size_t chNodeTo = (*m_nodeTo)[stage];
        DTensor<real_t> B(m_data.inputDynamics(), m_matAxis, chNodeFr, chNodeTo);
        DTensor<real_t> Btr = B.tr();
        DTensor<real_t> q(*m_d_q, m_matAxis, chNodeFr, chNodeTo);
        DTensor<real_t> Bq = Btr * *m_d_q;
        size_t nodeFr = (*m_nodeFrom)[stage-1];
        size_t nodeTo = (*m_nodeTo)[stage-1];
        for (size_t node=nodeFr; node<=nodeTo; node++) {
            DTensor<real_t> dAtParent(*m_d_d, m_matAxis, node, node);
            for (size_t child=(*m_childFrom)[node]; child<=(*m_childTo)[node]; child++) {
                DTensor<real_t> BqAtChild(Bq, m_matAxis, child, child);
                dAtParent += BqAtChild;
            }
        }
        DTensor<real_t> dAtStage(*m_d_d, m_matAxis, nodeFr, nodeTo);
        DTensor<real_t> uAtStage(*m_d_u, m_matAxis, nodeFr, nodeTo);
        dAtStage *= -1.;
        dAtStage += uAtStage;
        std::cout << "d at stage " << stage - 1 << dAtStage;
    }
}

void Cache::projectOnKernel() {

}

void Cache::vanillaCp(std::vector<real_t> initState, std::vector<real_t> *previousSolution) {
    /* Set initial state */
    if (initState.size() != m_data.numStates()) {
        std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                  << " but given " << initState.size() << " states" << "\n";
        throw std::invalid_argument("Incorrect dimension of initial state");
    }
    DTensor<real_t> slicePrim(*m_d_prim, 0, 0, m_data.numStates() - 1);
    slicePrim.upload(initState);
    /* Load previous solution if given */
    if (previousSolution) m_d_prim->upload(*previousSolution);
    /* Run CP algo */
    for (size_t i = 0; i < m_maxIters; i++) {
        cpIter();
        /** compute error */
        /** check error */
        if ((*m_d_cacheError)(i) <= m_tol) {
            m_countIterations = i;
            break;
        }
    }
}

/**
 * Compute one (1) iteration of vanilla CP algorithm, nothing more.
 */
void Cache::cpIter() {
    projectOnDynamics();
    projectOnKernel();
    /** update z_bar */
    /** update n_bar */
    /** update z */
    /** update n */
}

void Cache::print() {
    std::cout << "Tolerance: " << m_tol << "\n";
    std::cout << "Num iterations: " << m_countIterations << " of " << m_maxIters << "\n";
    std::cout << "Primal (from device): " << m_d_prim->tr();
}


#endif
