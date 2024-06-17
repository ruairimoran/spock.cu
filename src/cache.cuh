#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


template<typename T>
__global__ void k_setToZero(T *vec, size_t n);


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
TEMPLATE_WITH_TYPE_T
class Cache {

private:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
    T m_tol = 0;
    size_t m_maxIters = 0;
    size_t m_countIterations = 0;
    size_t m_matAxis = 2;
    size_t m_primSize = 0;
    size_t m_sizeU = 0;  ///< Inputs of all nonleaf nodes
    size_t m_sizeX = 0;  ///< States of all nodes
    size_t m_sizeY = 0;  ///< Y for all nonleaf nodes
    size_t m_sizeT = 0;  ///< T for all child nodes
    size_t m_sizeS = 0;  ///< S for all child nodes
    std::unique_ptr<DTensor<T>> m_d_prim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_primPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_u = nullptr;
    std::unique_ptr<DTensor<T>> m_d_x = nullptr;
    std::unique_ptr<DTensor<T>> m_d_y = nullptr;
    std::unique_ptr<DTensor<T>> m_d_t = nullptr;
    std::unique_ptr<DTensor<T>> m_d_s = nullptr;
    size_t m_dualSize = 0;
    std::unique_ptr<DTensor<T>> m_d_dual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_cacheError = nullptr;
    /* Other */
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_d = nullptr;

    /**
     * Private methods
     */
    void reshapePrimal();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree<T> &tree, ProblemData<T> &data, T tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        /* Sizes */
        m_sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
        m_sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
        m_sizeY = m_tree.numNonleafNodes() * m_tree.numEvents();  ///< Y for all nonleaf nodes
        m_sizeT = m_tree.numNodes();  ///< T for all child nodes
        m_sizeS = m_tree.numNodes();  ///< S for all child nodes
        m_primSize = m_sizeU + m_sizeX + m_sizeY + m_sizeT + m_sizeS;
        /* Allocate memory on device */
        m_d_prim = std::make_unique<DTensor<T>>(m_primSize, true);
        m_d_primPrev = std::make_unique<DTensor<T>>(m_primSize, true);
        m_d_dual = std::make_unique<DTensor<T>>(m_dualSize, true);
        m_d_dualPrev = std::make_unique<DTensor<T>>(m_dualSize, true);
        m_d_cacheError = std::make_unique<DTensor<T>>(m_maxIters, true);
        m_d_q = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        /* Slice primal */
        reshapePrimal();
    }

    ~Cache() {}

    /**
     * Public methods
     */
    void initialiseState(std::vector<T> &initState);

    void projectOnDynamics();

    void projectOnKernel();

    void cpIter();

    void vanillaCp(std::vector<T> &initState, std::vector<T> *previousSolution = nullptr);

    /**
     * Getters
     */
    size_t solutionSize() { return m_primSize; }

    DTensor<T> &solution() { return *m_d_prim; }

    DTensor<T> &inputs() { return *m_d_u; }

    DTensor<T> &states() { return *m_d_x; }

    /**
     * Debugging
     */
    void print();
};

template<typename T>
void Cache<T>::reshapePrimal() {
    size_t rowAxis = 0;
    size_t start = 0;
    m_d_u = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeU - 1);
    m_d_u->reshape(m_data.numInputs(), 1, m_tree.numNonleafNodes());
    start += m_sizeU;
    m_d_x = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeX - 1);
    m_d_x->reshape(m_data.numStates(), 1, m_tree.numNodes());
    start += m_sizeX;
    m_d_y = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeY - 1);
    m_d_y->reshape(m_tree.numEvents(), 1, m_tree.numNonleafNodes());
    start += m_sizeY;
    m_d_t = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeT - 1);
    m_d_t->reshape(1, 1, m_tree.numNodes());
    start += m_sizeT;
    m_d_s = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeS - 1);
    m_d_s->reshape(1, 1, m_tree.numNodes());
}

template<typename T>
void Cache<T>::initialiseState(std::vector<T> &initState) {
    /* Set initial state */
    if (initState.size() != m_data.numStates()) {
        std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                  << " but given " << initState.size() << " states" << "\n";
        throw std::invalid_argument("Incorrect dimension of initial state");
    }
    DTensor<T> firstState(*m_d_x, m_matAxis, 0, 0);
    firstState.upload(initState);
}

template<typename T>
void Cache<T>::projectOnDynamics() {
    /* Reset d and q */
    k_setToZero<<<numBlocks(m_d_d->numEl()), TPB>>>(m_d_d->raw(), m_d_d->numEl());
    k_setToZero<<<numBlocks(m_d_q->numEl()), TPB>>>(m_d_q->raw(), m_d_q->numEl());
    /* Set first q */
    DTensor<T> statesAtLastStage(*m_d_x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    statesAtLastStage *= -1.;
    DTensor<T> q0(*m_d_q, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    statesAtLastStage.deviceCopyTo(q0);
    std::cout << "-x @ last stage: " << statesAtLastStage;
    std::cout << "q @ last stage: " << q0;
    /* Solve for next d */
    size_t horizon = m_tree.numStages() - 1;
    for (size_t t = 1; t < m_tree.numStages(); t++) {
        size_t stage = horizon - t;
        size_t nodeFr = m_tree.nodeFrom()[stage];
        size_t nodeTo = m_tree.nodeTo()[stage];
        size_t chStage = stage + 1;
        size_t chNodeFr = m_tree.nodeFrom()[chStage];
        size_t chNodeTo = m_tree.nodeTo()[chStage];
        DTensor<T> BAtChStage(m_data.inputDynamics(), m_matAxis, chNodeFr, chNodeTo);
        std::cout << "B @ stage " << chStage << ": " << BAtChStage;
        DTensor<T> Btr = BAtChStage.tr();  // this can be done offline
        DTensor<T> qAtChStage(*m_d_q, m_matAxis, chNodeFr, chNodeTo);
        std::cout << "q @ stage " << chStage << ": " << qAtChStage;
        DTensor<T> Bq = Btr * *m_d_q;
        std::cout << "Bq @ stage " << chStage << ": " << Bq; // here
        for (size_t node = nodeFr; node <= nodeTo; node++) {
            DTensor<T> dAtAncNode(*m_d_d, m_matAxis, node, node);
            for (size_t ch = m_tree.childFrom()[node]; ch <= m_tree.childTo()[node]; ch++) {
                DTensor<T> BqAtChildNode(Bq, m_matAxis, ch, ch);
                dAtAncNode += BqAtChildNode;
            }
            std::cout << "d1 @ node " << node << ": " << dAtAncNode;
        }
        DTensor<T> dAtStage(*m_d_d, m_matAxis, nodeFr, nodeTo);
        DTensor<T> uAtStage(*m_d_u, m_matAxis, nodeFr, nodeTo);
        dAtStage *= -1.;
        dAtStage += uAtStage;
        std::cout << "d2 @ stage " << stage << ": " << dAtStage;
        m_data.choleskyBatch()[stage]->solve(dAtStage);
        std::cout << "d3 @ stage " << stage << ": " << dAtStage;
        /* Solve for next q */
        DTensor<T> K(m_data.K(), m_matAxis, nodeFr, nodeTo);
        DTensor<T> Ktr = K.tr();
        DTensor<T> inputsAtStage(*m_d_u, m_matAxis, nodeFr, nodeTo);
        DTensor<T> dMinusInputs = dAtStage - inputsAtStage;
        DTensor<T> Kdu = Ktr * dMinusInputs;
        DTensor<T> statesAtStage(*m_d_x, m_matAxis, nodeFr, nodeTo);
        Kdu -= statesAtStage;
        for (size_t node = nodeFr; node <= nodeTo; node++) {
            DTensor<T> dAtParent(*m_d_d, m_matAxis, node, node);
            DTensor<T> qAtParent(*m_d_q, m_matAxis, node, node);
            qAtParent = Kdu;
            for (size_t child = m_tree.childFrom()[node]; child <= m_tree.childTo()[node]; child++) {
                DTensor<T> APBAtChild(m_data.APB(), m_matAxis, child, child);
                DTensor<T> AAtChild(m_data.dynamicsSumTr(), m_matAxis, child, child);
                DTensor<T> qAtChild(*m_d_q, m_matAxis, child, child);
                DTensor<T> APBd = APBAtChild * dAtParent;
                DTensor<T> Aq = AAtChild * qAtChild;
                qAtParent += APBd + Aq;
            }
//            std::cout << "q @ node " << node << ": " << qAtParent;
        }
    }
    /* State has been initialised, now compute control actions */
    for (size_t stage = 0; stage < m_tree.numStages() - 1; stage++) {
        size_t nodeFr = m_tree.nodeFrom()[stage];
        size_t nodeTo = m_tree.nodeTo()[stage];
        /* Compute control actions */
        DTensor<T> KAtStage(m_data.K(), m_matAxis, nodeFr, nodeTo);
//        std::cout << "K @ stage " << stage << ": " << KAtStage;
        DTensor<T> xAtStage(*m_d_x, m_matAxis, nodeFr, nodeTo);
//        std::cout << "x @ stage " << stage << ": " << xAtStage;
        DTensor<T> dAtStage(*m_d_d, m_matAxis, nodeFr, nodeTo);
        std::cout << "read d @ stage " << stage << ": " << dAtStage;
        DTensor<T> uAtStage(*m_d_u, m_matAxis, nodeFr, nodeTo);
        DTensor<T> KxdAtStage = KAtStage * xAtStage;
        KxdAtStage += dAtStage;
        KxdAtStage.deviceCopyTo(uAtStage);
//        std::cout << "u @ stage " << stage << ": " << uAtStage;
        /* Compute next states */
        for (size_t node = nodeFr; node <= nodeTo; node++) {
            DTensor<T> xAtParent(*m_d_x, m_matAxis, node, node);
            DTensor<T> uAtParent(*m_d_u, m_matAxis, node, node);
            for (size_t child = m_tree.childFrom()[node]; child <= m_tree.childTo()[node]; child++) {
                DTensor<T> xAtChild(*m_d_x, m_matAxis, child, child);
                DTensor<T> AAtChild(m_data.stateDynamics(), m_matAxis, child, child);
                DTensor<T> BAtChild(m_data.inputDynamics(), m_matAxis, child, child);
                DTensor<T> Ax = AAtChild * xAtParent;
                DTensor<T> Bu = BAtChild * uAtParent;
                DTensor<T> AxPlusBu = Ax + Bu;
                AxPlusBu.deviceCopyTo(xAtChild);
//                std::cout << "x @ node " << node << ": " << xAtChild;
            }
        }
    }
}

template<typename T>
void Cache<T>::projectOnKernel() {

}

template<typename T>
void Cache<T>::vanillaCp(std::vector<T> &initState, std::vector<T> *previousSolution) {
    initialiseState(initState);
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
template<typename T>
void Cache<T>::cpIter() {
    projectOnDynamics();
    projectOnKernel();
    /** update z_bar */
    /** update n_bar */
    /** update z */
    /** update n */
}

template<typename T>
void Cache<T>::print() {
    std::cout << "Tolerance: " << m_tol << "\n";
    std::cout << "Num iterations: " << m_countIterations << " of " << m_maxIters << "\n";
    std::cout << "Primal (from device): " << m_d_prim->tr();
}


#endif
