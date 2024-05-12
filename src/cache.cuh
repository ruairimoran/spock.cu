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
    size_t m_primSize = 0;
    size_t m_sizeX = 0;  ///< States of all nodes
    size_t m_sizeU = 0;  ///< Inputs of all nonleaf nodes
    size_t m_sizeY = 0;  ///< Y for all nonleaf nodes
    size_t m_sizeT = 0;  ///< T for all child nodes
    size_t m_sizeS = 0;  ///< S for all child nodes
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
    void breakPrimal();
    void rebuildPrimal();
    void projectOnDynamics();
    void projectOnKernel();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree &tree, ProblemData &data, real_t tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        /* Sizes */
        m_sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
        m_sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
        m_sizeY = m_tree.numNonleafNodes() * m_tree.numEvents();  ///< Y for all nonleaf nodes
        m_sizeT = m_tree.numNodes();  ///< T for all child nodes
        m_sizeS = m_tree.numNodes();  ///< S for all child nodes
        m_primSize = m_sizeX + m_sizeU + m_sizeY + m_sizeT + m_sizeS;
        /* Allocate memory on device */
        m_d_prim = std::make_unique<DTensor<real_t>>(m_primSize, true);
        m_d_primPrev = std::make_unique<DTensor<real_t>>(m_primSize, true);
        m_d_dual = std::make_unique<DTensor<real_t>>(m_dualSize, true);
        m_d_dualPrev = std::make_unique<DTensor<real_t>>(m_dualSize, true);
        m_d_cacheError = std::make_unique<DTensor<real_t>>(m_maxIters, true);
        m_d_q = std::make_unique<DTensor<real_t>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<real_t>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        /* Slice primal */
        breakPrimal();
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

void Cache::breakPrimal() {
    size_t rowAxis = 0;
    size_t start = 0;
    DTensor<real_t> sliceX(*m_d_prim, rowAxis, start, m_sizeX - 1);
    m_d_x = std::make_unique<DTensor<real_t>>(m_data.numStates(), 1, m_tree.numNodes());
    sliceX.deviceCopyTo(*m_d_x);
    start += m_sizeX;
    DTensor<real_t> sliceU(*m_d_prim, rowAxis, start, start + m_sizeU - 1);
    m_d_u = std::make_unique<DTensor<real_t>>(m_data.numInputs(), 1, m_tree.numNonleafNodes());
    sliceU.deviceCopyTo(*m_d_u);
    start += m_sizeU;
    DTensor<real_t> sliceY(*m_d_prim, rowAxis, start, start + m_sizeY - 1);
    m_d_y = std::make_unique<DTensor<real_t>>(m_tree.numEvents(), 1, m_tree.numNonleafNodes());
    sliceY.deviceCopyTo(*m_d_y);
    start += m_sizeY;
    DTensor<real_t> sliceT(*m_d_prim, rowAxis, start, start + m_sizeT - 1);
    m_d_t = std::make_unique<DTensor<real_t>>(1, 1, m_tree.numNodes());
    sliceT.deviceCopyTo(*m_d_t);
    start += m_sizeT;
    DTensor<real_t> sliceS(*m_d_prim, rowAxis, start, start + m_sizeS - 1);
    m_d_s = std::make_unique<DTensor<real_t>>(1, 1, m_tree.numNodes());
    sliceS.deviceCopyTo(*m_d_s);
}

void Cache::rebuildPrimal() {
    size_t rowAxis = 0;
    size_t start = 0;
    DTensor<real_t> sliceX(*m_d_prim, rowAxis, start, m_sizeX - 1);
    m_d_x->deviceCopyTo(sliceX);
    start += m_sizeX;
    DTensor<real_t> sliceU(*m_d_prim, rowAxis, start, start + m_sizeU - 1);
    m_d_u->deviceCopyTo(sliceU);
    start += m_sizeU;
    DTensor<real_t> sliceY(*m_d_prim, rowAxis, start, start + m_sizeY - 1);
    m_d_y->deviceCopyTo(sliceY);
    start += m_sizeY;
    DTensor<real_t> sliceT(*m_d_prim, rowAxis, start, start + m_sizeT - 1);
    m_d_t->deviceCopyTo(sliceT);
    start += m_sizeT;
    DTensor<real_t> sliceS(*m_d_prim, rowAxis, start, start + m_sizeS - 1);
    m_d_s->deviceCopyTo(sliceS);
}

void Cache::projectOnDynamics() {
    *m_d_x *= -1.;
    m_d_x->deviceCopyTo(*m_d_q);
    for (size_t stagePlusOne=m_tree.numStages()-1; stagePlusOne>0; stagePlusOne--) {
        size_t stage = stagePlusOne - 1;
        size_t chNodeFr = (*m_tree.nodeFromHost())[stagePlusOne];
        size_t chNodeTo = (*m_tree.nodeToHost())[stagePlusOne];
        DTensor<real_t> B(m_data.inputDynamics(), m_matAxis, chNodeFr, chNodeTo);
        DTensor<real_t> Btr = B.tr();
        DTensor<real_t> q(*m_d_q, m_matAxis, chNodeFr, chNodeTo);
        DTensor<real_t> Bq = Btr * *m_d_q;
        size_t nodeFr = (*m_tree.nodeFromHost())[stage];
        size_t nodeTo = (*m_tree.nodeToHost())[stage];
        for (size_t node=nodeFr; node<=nodeTo; node++) {
            DTensor<real_t> dAtParent(*m_d_d, m_matAxis, node, node);
            for (size_t child=(*m_tree.childFromHost())[node]; child<=(*m_tree.childToHost())[node]; child++) {
                DTensor<real_t> BqAtChild(Bq, m_matAxis, child, child);
                dAtParent += BqAtChild;
            }
        }
        DTensor<real_t> dAtStage(*m_d_d, m_matAxis, nodeFr, nodeTo);
        DTensor<real_t> uAtStage(*m_d_u, m_matAxis, nodeFr, nodeTo);
        dAtStage *= -1.;
        dAtStage += uAtStage;
        m_data.choleskyBatch()[stage]->solve(dAtStage);
        std::cout << "d at stage " << stage << "\n" << dAtStage;
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
    rebuildPrimal();
}

void Cache::print() {
    std::cout << "Tolerance: " << m_tol << "\n";
    std::cout << "Num iterations: " << m_countIterations << " of " << m_maxIters << "\n";
    std::cout << "Primal (from device): " << m_d_prim->tr();
}


#endif
