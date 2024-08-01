#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"
#include "operator.cuh"


template<typename T>
__global__ void k_setToZero(T *vec, size_t n);

template<typename T>
class CacheTestData;

template<typename T>
void testInitialisingState(CacheTestData<T> &);

template<typename T>
void testDynamicsProjectionOnline(CacheTestData<T> &, T);

template<typename T>
void testKernelProjectionOnline(CacheTestData<T> &, T);

template<typename T>
void testKernelProjectionOnlineOrthogonality(CacheTestData<T> &, T);

template<typename T>
class OperatorTestData;

template<typename T>
void testOperator(OperatorTestData<T> &, T);

template<typename T>
void testAdjoint(OperatorTestData<T> &, T);


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
TEMPLATE_WITH_TYPE_T
class Cache {
protected:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
    LinearOperator<T> m_L = LinearOperator<T>(m_tree, m_data);  ///< Linear operator and its adjoint
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
    size_t m_sizeI = 0;
    size_t m_sizeII = 0;
    size_t m_sizeIII = 0;
    size_t m_sizeIV = 0;
    size_t m_sizeV = 0;
    size_t m_sizeVI = 0;
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
    std::unique_ptr<DTensor<T>> m_d_i = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ii = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iii = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iv = nullptr;
    std::unique_ptr<DTensor<T>> m_d_v = nullptr;
    std::unique_ptr<DTensor<T>> m_d_vi = nullptr;
    std::unique_ptr<DTensor<T>> m_d_cacheError = nullptr;
    /* Other */
    std::unique_ptr<DTensor<T>> m_d_initState = nullptr;
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_d = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ytsSizeWorkspace = nullptr;

    /**
     * Protected methods
     */
    void reshapePrimal();

    void reshapeDual();

    void setRootState();

    void initialiseState(std::vector<T> &initState);

    void projectOnDynamics();

    void projectOnKernels();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree<T> &tree, ProblemData<T> &data, T tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        /* Sizes */
        m_sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
        m_sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
        m_sizeY = m_tree.numNonleafNodes() * m_data.yDim();  ///< Y for all nonleaf nodes
        m_sizeT = m_tree.numNodes();  ///< T for all child nodes
        m_sizeS = m_tree.numNodes();  ///< S for all child nodes
        m_primSize = m_sizeU + m_sizeX + m_sizeY + m_sizeT + m_sizeS;
        m_sizeI = m_tree.numNonleafNodes() * m_data.yDim();
        m_sizeII = m_tree.numNonleafNodes();
        m_sizeIII = m_tree.numNonleafNodes() * m_data.numStatesAndInputs();  // Might need to change for non-rectangles
        m_sizeIV = m_tree.numNodes() * (m_data.numStatesAndInputs() + 2);
        m_sizeV = m_tree.numLeafNodes() * m_data.numStates();
        m_sizeVI = m_tree.numLeafNodes() * (m_data.numStates() + 2);
        m_dualSize = m_sizeI + m_sizeII + m_sizeIII + m_sizeIV + m_sizeV + m_sizeVI;
        /* Allocate memory on device */
        m_d_prim = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_primPrev = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_dual = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_dualPrev = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_cacheError = std::make_unique<DTensor<T>>(m_maxIters, 1, 1, true);
        m_d_initState = std::make_unique<DTensor<T>>(m_data.numStates(), 1, 1, true);
        m_d_q = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_xSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_uSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        m_d_xuSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs(), 1, m_tree.numNodes(), true);
        m_d_ytsSizeWorkspace = std::make_unique<DTensor<T>>(m_data.nullDim(), 1, m_tree.numNonleafNodes(), true);
        /* Slice and reshape primal and dual */
        reshapePrimal();
        reshapeDual();
    }

    ~Cache() = default;

    /**
     * Public methods
     */
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
     * Test functions. As a friend, they can access protected members.
     */
    friend void testInitialisingState<>(CacheTestData<T> &);

    friend void testDynamicsProjectionOnline<>(CacheTestData<T> &, T);

    friend void testKernelProjectionOnline<>(CacheTestData<T> &, T);

    friend void testKernelProjectionOnlineOrthogonality<>(CacheTestData<T> &, T);

    friend void testOperator<>(OperatorTestData<T> &, T);

    friend void testAdjoint<>(OperatorTestData<T> &, T);

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
    m_d_y->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    start += m_sizeY;
    m_d_t = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeT - 1);
    m_d_t->reshape(1, 1, m_tree.numNodes());
    start += m_sizeT;
    m_d_s = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeS - 1);
    m_d_s->reshape(1, 1, m_tree.numNodes());
}

template<typename T>
void Cache<T>::reshapeDual() {
    size_t rowAxis = 0;
    size_t start = 0;
    m_d_i = std::make_unique<DTensor<T>>(*m_d_dual, rowAxis, start, start + m_sizeI - 1);
    m_d_i->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    start += m_sizeI;
    m_d_ii = std::make_unique<DTensor<T>>(*m_d_dual, rowAxis, start, start + m_sizeII - 1);
    m_d_ii->reshape(1, 1, m_tree.numNonleafNodes());
    start += m_sizeII;
    m_d_iii = std::make_unique<DTensor<T>>(*m_d_dual, rowAxis, start, start + m_sizeIII - 1);
    m_d_iii->reshape(m_data.numStatesAndInputs(), 1, m_tree.numNonleafNodes());
    start += m_sizeIII;
    m_d_iv = std::make_unique<DTensor<T>>(*m_d_dual, rowAxis, start, start + m_sizeIV - 1);
    m_d_iv->reshape(m_data.numStatesAndInputs() + 2, 1, m_tree.numNodes());
    start += m_sizeIV;
    m_d_v = std::make_unique<DTensor<T>>(*m_d_dual, rowAxis, start, start + m_sizeV - 1);
    m_d_v->reshape(m_data.numStates(), 1, m_tree.numLeafNodes());
    start += m_sizeV;
    m_d_vi = std::make_unique<DTensor<T>>(*m_d_dual, rowAxis, start, start + m_sizeVI - 1);
    m_d_vi->reshape(m_data.numStates() + 2, 1, m_tree.numLeafNodes());
}

template<typename T>
void Cache<T>::initialiseState(std::vector<T> &initState) {
    /* Set initial state */
    if (initState.size() != m_data.numStates()) {
        std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                  << " but given " << initState.size() << " states" << "\n";
        throw std::invalid_argument("[initialiseState] Incorrect dimension of initial state");
    }
    m_d_initState->upload(initState);
    setRootState();
}

template<typename T>
void Cache<T>::setRootState() {
    DTensor<T> firstState(*m_d_x, m_matAxis, 0, 0);
    m_d_initState->deviceCopyTo(firstState);
}

template<typename T>
void Cache<T>::projectOnDynamics() {
    /*
     * Set first q
     */
    DTensor<T> x_LastStage(*m_d_x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    x_LastStage *= -1.;
    DTensor<T> q_LastStage(*m_d_q, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    x_LastStage.deviceCopyTo(q_LastStage);
    /*
     * Solve for all d at current stage
     */
    size_t horizon = m_tree.numStages() - 1;
    for (size_t t = 1; t < m_tree.numStages(); t++) {
        size_t stage = horizon - t;  // Current stage (reverse from N-1 to 0)
        size_t stageFr = m_tree.stageFrom()[stage];  // First node of current stage
        size_t stageTo = m_tree.stageTo()[stage];  // Last node of current stage
        size_t chStage = stage + 1;  // Stage of children of current stage
        size_t chStageFr = m_tree.stageFrom()[chStage];  // First node of child stage
        size_t chStageTo = m_tree.stageTo()[chStage];  // Last node of child stage
        size_t maxCh = m_tree.childMax()[stage];  // Max number of children of any node at current stage
        /* Compute `Bq` at every child of current stage */
        DTensor<T> Btr_ChStage(m_data.inputDynamicsTr(), m_matAxis, chStageFr, chStageTo);
        DTensor<T> q_ChStage(*m_d_q, m_matAxis, chStageFr, chStageTo);
        DTensor<T> Bq_ChStage(*m_d_uSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        Bq_ChStage.addAB(Btr_ChStage, q_ChStage);
        /* Copy into `d` the first child `Bq` of each node (every nonleaf node has at least one child) */
        for (size_t node = stageFr; node <= stageTo; node++) {
            size_t ch = m_tree.childFrom()[node];
            DTensor<T> Bq_ChNode(*m_d_uSizeWorkspace, m_matAxis, ch, ch);
            DTensor<T> Bq_Node(*m_d_d, m_matAxis, node, node);
            Bq_ChNode.deviceCopyTo(Bq_Node);
        }
        /* Add to `d` remaining child `Bq` of each node (zeros added if child does not exist) */
        DTensor<T> d_Stage(*m_d_d, m_matAxis, stageFr, stageTo);
        for (size_t chIdx = 1; chIdx < maxCh; chIdx++) {  // Index of child of every node at current stage
            for (size_t node = stageFr; node <= stageTo; node++) {
                size_t ch = m_tree.childFrom()[node] + chIdx;
                if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Bq_ChStage`
                    DTensor<T> Bq_ChNode(*m_d_uSizeWorkspace, m_matAxis, ch, ch);
                    DTensor<T> Bq_Node(*m_d_uSizeWorkspace, m_matAxis, node, node);
                    Bq_ChNode.deviceCopyTo(Bq_Node);
                }
            }
            DTensor<T> d_Add(*m_d_uSizeWorkspace, m_matAxis, stageFr, stageTo);
            d_Stage += d_Add;
        }
        /* Subtract d from u in place */
        DTensor<T> u_Stage(*m_d_u, m_matAxis, stageFr, stageTo);
        d_Stage *= -1.;
        d_Stage += u_Stage;
        /* Use Cholesky decomposition for final step of computing all d at current stage */
        m_data.choleskyBatch()[stage]->solve(d_Stage);
        /*
         * Solve for all q at current stage
         */
        /* Compute APBdAq_ChStage = A(PBd+q) for each node at child stage. A = (A+B@K).tr */
        DTensor<T> APB_ChStage(m_data.APB(), m_matAxis, chStageFr, chStageTo);
        DTensor<T> ABKtr_ChStage(m_data.dynamicsSumTr(), m_matAxis, chStageFr, chStageTo);
        for (size_t node = stageFr; node <= stageTo; node++) {
            DTensor<T> d_Node(*m_d_d, m_matAxis, node, node);
            for (size_t ch = m_tree.childFrom()[node]; ch <= m_tree.childTo()[node]; ch++) {
                DTensor<T> d_ExpandedChNode(*m_d_uSizeWorkspace, m_matAxis, ch, ch);
                d_Node.deviceCopyTo(d_ExpandedChNode);
            }
        }
        DTensor<T> q_SumChStage(*m_d_xSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        DTensor<T> d_ExpandedChStage(*m_d_uSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        q_SumChStage.addAB(APB_ChStage, d_ExpandedChStage);
        q_SumChStage.addAB(ABKtr_ChStage, q_ChStage, 1., 1.);
        /* Copy into `q` the first child `APBdAq` of each node (every nonleaf node has at least one child) */
        for (size_t node = stageFr; node <= stageTo; node++) {
            size_t ch = m_tree.childFrom()[node];
            DTensor<T> APBdAq_ChNode(*m_d_xSizeWorkspace, m_matAxis, ch, ch);
            DTensor<T> APBdAq_Node(*m_d_q, m_matAxis, node, node);
            APBdAq_ChNode.deviceCopyTo(APBdAq_Node);
        }
        /* Add to `q` remaining child `APBdAq` of each node (zeros added if child does not exist) */
        DTensor<T> q_Stage(*m_d_q, m_matAxis, stageFr, stageTo);
        for (size_t chOfEachNode = 1; chOfEachNode < maxCh; chOfEachNode++) {
            for (size_t node = stageFr; node <= stageTo; node++) {
                size_t ch = m_tree.childFrom()[node] + chOfEachNode;
                if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Bq_ChStage`
                    DTensor<T> APBdAq_ChNode(*m_d_xSizeWorkspace, m_matAxis, ch, ch);
                    DTensor<T> APBdAq_Node(*m_d_xSizeWorkspace, m_matAxis, node, node);
                    APBdAq_ChNode.deviceCopyTo(APBdAq_Node);
                }
            }
            DTensor<T> q_Add(*m_d_xSizeWorkspace, m_matAxis, stageFr, stageTo);
            q_Stage += q_Add;
        }
        /* Compute Kdux = K.tr(d-u)@x for each node at current stage and add to `q` */
        DTensor<T> du_Stage(*m_d_uSizeWorkspace, m_matAxis, stageFr, stageTo);
        d_Stage.deviceCopyTo(du_Stage);
        du_Stage -= u_Stage;
        DTensor<T> Ktr_Stage(m_data.KTr(), m_matAxis, stageFr, stageTo);
        q_Stage.addAB(Ktr_Stage, du_Stage, 1., 1.);
        DTensor<T> x_Stage(*m_d_x, m_matAxis, stageFr, stageTo);
        q_Stage -= x_Stage;
    }
    /*
     * Set initial state
     */
    setRootState();
    for (size_t stage = 0; stage < m_tree.numStages() - 1; stage++) {
        size_t stageFr = m_tree.stageFrom()[stage];
        size_t stageTo = m_tree.stageTo()[stage];
        size_t chStage = stage + 1;  // Stage of children of current stage
        size_t chStageFr = m_tree.stageFrom()[chStage];  // First node of child stage
        size_t chStageTo = m_tree.stageTo()[chStage];  // Last node of child stage
        /*
         * Compute next control action
         */
        DTensor<T> uAtStage(*m_d_u, m_matAxis, stageFr, stageTo);
        DTensor<T> KAtStage(m_data.K(), m_matAxis, stageFr, stageTo);
        DTensor<T> xAtStage(*m_d_x, m_matAxis, stageFr, stageTo);
        DTensor<T> dAtStage(*m_d_d, m_matAxis, stageFr, stageTo);
        uAtStage.addAB(KAtStage, xAtStage);
        uAtStage += dAtStage;
        /*
         * Compute child states
         */
        /* Fill `xu` */
        for (size_t node = stageFr; node <= stageTo; node++) {
            DTensor<T> x_Node(*m_d_x, m_matAxis, node, node);
            DTensor<T> u_Node(*m_d_u, m_matAxis, node, node);
            for (size_t ch = m_tree.childFrom()[node]; ch <= m_tree.childTo()[node]; ch++) {
                DTensor<T> xu_ChNode(*m_d_xuSizeWorkspace, m_matAxis, ch, ch);
                DTensor<T> xu_sliceX(xu_ChNode, 0, 0, m_data.numStates() - 1);
                DTensor<T> xu_sliceU(xu_ChNode, 0, m_data.numStates(), m_data.numStatesAndInputs() - 1);
                x_Node.deviceCopyTo(xu_sliceX);
                u_Node.deviceCopyTo(xu_sliceU);
            }
        }
        DTensor<T> x_ChStage(*m_d_x, m_matAxis, chStageFr, chStageTo);
        DTensor<T> AB_ChStage(m_data.stateInputDynamics(), m_matAxis, chStageFr, chStageTo);
        DTensor<T> xu_ChStage(*m_d_xuSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        x_ChStage.addAB(AB_ChStage, xu_ChStage);
    }
}

template<typename T>
void Cache<T>::projectOnKernels() {
    /**
     * Project on kernel of every node of tree at once
     */
    /* Gather vec[i] = (y_i, t[ch(i)], s[ch(i)]) for all nonleaf nodes */
    for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
        size_t chFr = m_tree.childFrom()[node];
        size_t chTo = m_tree.childTo()[node];
        size_t numCh = m_tree.numChildren()[node];
        DTensor<T> y(*m_d_y, m_matAxis, node, node);
        DTensor<T> t(*m_d_t, m_matAxis, chFr, chTo);
        DTensor<T> s(*m_d_s, m_matAxis, chFr, chTo);
        DTensor<T> nodeStore(*m_d_ytsSizeWorkspace, m_matAxis, node, node);
        DTensor<T> yStore(nodeStore, 0, 0, m_data.yDim() - 1);
        DTensor<T> tStore(nodeStore, 0, m_data.yDim(), m_data.yDim() + numCh - 1);
        DTensor<T> sStore(nodeStore, 0, m_data.yDim() + m_tree.numEvents(),
                          m_data.yDim() + m_tree.numEvents() + numCh - 1);
        tStore.reshape(1, 1, numCh);
        sStore.reshape(1, 1, numCh);
        y.deviceCopyTo(yStore);
        t.deviceCopyTo(tStore);
        s.deviceCopyTo(sStore);
    }
    /* Project onto nullspace in place */
    m_d_ytsSizeWorkspace->addAB(m_data.nullspaceProj(), *m_d_ytsSizeWorkspace);
    /* Disperse vec[i] = (y_i, t[ch(i)], s[ch(i)]) for all nonleaf nodes */
    for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
        size_t chFr = m_tree.childFrom()[node];
        size_t chTo = m_tree.childTo()[node];
        size_t numCh = m_tree.numChildren()[node];
        DTensor<T> y(*m_d_y, m_matAxis, node, node);
        DTensor<T> t(*m_d_t, m_matAxis, chFr, chTo);
        DTensor<T> s(*m_d_s, m_matAxis, chFr, chTo);
        DTensor<T> nodeStore(*m_d_ytsSizeWorkspace, m_matAxis, node, node);
        DTensor<T> yStore(nodeStore, 0, 0, m_data.yDim() - 1);
        DTensor<T> tStore(nodeStore, 0, m_data.yDim(), m_data.yDim() + numCh - 1);
        DTensor<T> sStore(nodeStore, 0, m_data.yDim() + m_tree.numEvents(),
                          m_data.yDim() + m_tree.numEvents() + numCh - 1);
        tStore.reshape(1, 1, numCh);
        sStore.reshape(1, 1, numCh);
        yStore.deviceCopyTo(y);
        tStore.deviceCopyTo(t);
        sStore.deviceCopyTo(s);
    }
}

/**
 * Compute one (1) iteration of vanilla CP algorithm, nothing more.
 */
template<typename T>
void Cache<T>::cpIter() {
    projectOnDynamics();
    projectOnKernels();
    m_L.op(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    m_L.adj(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    /** update z_bar */
    /** update n_bar */
    /** update z */
    /** update n */
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

template<typename T>
void Cache<T>::print() {
    std::cout << "Tolerance: " << m_tol << "\n";
    std::cout << "Num iterations: " << m_countIterations << " of " << m_maxIters << "\n";
    std::cout << "Primal (from device): " << m_d_prim->tr();
}


#endif
