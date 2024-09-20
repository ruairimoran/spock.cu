#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"
#include "operator.cuh"
#include "projections.cuh"
#include <chrono>


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

template<typename T>
void testComputeErrors(CacheTestData<T> &, T);


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
    bool m_detectInfeas = false;
    size_t m_countIterations = 0;
    T m_err = 0;
    size_t m_warmupIters = 0;
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
    std::unique_ptr<DTensor<T>> m_d_adjDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_adjDualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_u = nullptr;
    std::unique_ptr<DTensor<T>> m_d_x = nullptr;
    std::unique_ptr<DTensor<T>> m_d_y = nullptr;
    std::unique_ptr<DTensor<T>> m_d_t = nullptr;
    std::unique_ptr<DTensor<T>> m_d_s = nullptr;
    size_t m_dualSize = 0;
    std::unique_ptr<DTensor<T>> m_d_dual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_opPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_opPrimPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_i = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ii = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iii = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iv = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ivSoc = nullptr;
    std::unique_ptr<DTensor<T>> m_d_v = nullptr;
    std::unique_ptr<DTensor<T>> m_d_vi = nullptr;
    std::unique_ptr<DTensor<T>> m_d_viSoc = nullptr;
    /* Workspaces */
    std::unique_ptr<DTensor<T>> m_d_initState = nullptr;
    std::unique_ptr<DTensor<T>> m_d_primWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_d = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ytsSizeWorkspace = nullptr;
    /* Projections */
    std::unique_ptr<SocProjection<T>> m_socsNonleaf = nullptr;
    std::unique_ptr<SocProjection<T>> m_socsLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_socsNonleafHalves = nullptr;
    std::unique_ptr<DTensor<T>> m_d_socsLeafHalves = nullptr;
    std::unique_ptr<NonnegativeOrthantCone<T>> m_nnocNonleaf = nullptr;
    std::unique_ptr<Cartesian<T>> m_cartRisk = nullptr;
    std::unique_ptr<DTensor<T>> m_d_loBoundNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_hiBoundNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_loBoundLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_hiBoundLeaf = nullptr;
    /* Errors */
    std::unique_ptr<DTensor<T>> m_d_primErr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualErr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaDual = nullptr;
    std::vector<T> m_cacheError0;
    std::vector<T> m_cacheError1;
    std::vector<T> m_cacheError2;
    std::vector<T> m_cacheDeltaPrim;
    std::vector<T> m_cacheDeltaDual;
    std::vector<T> m_cacheNrmLtrDeltaDual;
    std::vector<T> m_cacheDistDeltaDual;
    std::vector<T> m_cacheSuppDeltaDual;
    std::vector<std::vector<T>> m_cachePrim;
    std::vector<std::vector<T>> m_cacheDual;

    /**
     * Protected methods
     */
    void reshapePrimalWorkspace();

    void reshapeDualWorkspace();

    void initialiseProjectable();

    void initialiseSizes();

    void setRootState();

    void initialiseState(std::vector<T> &initState);

    void proxRootS();

    void projectPrimalWorkspaceOnDynamics();

    void projectPrimalWorkspaceOnKernels();

    void translateSocs();

    void projectDualWorkspaceOnConstraints();

    void computeAdjDual();

    void modifyPrimal();

    void proximalPrimal();

    void modifyDual();

    void proximalDual();

    bool computeError(size_t);

    void printToJson();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree<T> &tree, ProblemData<T> &data, T tol, size_t maxIters, bool detectInfeas = false) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters), m_detectInfeas(detectInfeas) {
        /* Allocate memory on host */
        size_t cacheSize = m_maxIters - m_warmupIters;
        m_cacheError0 = std::vector<T>(cacheSize);
        m_cacheError1 = std::vector<T>(cacheSize);
        m_cacheError2 = std::vector<T>(cacheSize);
        m_cacheDeltaPrim = std::vector<T>(cacheSize);
        m_cacheDeltaDual = std::vector<T>(cacheSize);
        m_cacheNrmLtrDeltaDual = std::vector<T>(cacheSize);
        m_cacheDistDeltaDual = std::vector<T>(cacheSize);
        m_cacheSuppDeltaDual = std::vector<T>(cacheSize);
        /* Sizes */
        initialiseSizes();
        m_cachePrim = std::vector<std::vector<T>>(cacheSize, std::vector<T>(m_primSize));
        m_cacheDual = std::vector<std::vector<T>>(cacheSize, std::vector<T>(m_dualSize));
        /* Allocate memory on device */
        m_d_prim = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_primPrev = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_primWorkspace = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_dual = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_dualPrev = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_dualWorkspace = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_initState = std::make_unique<DTensor<T>>(m_data.numStates(), 1, 1, true);
        m_d_q = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_xSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_uSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        m_d_xuSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs(), 1, m_tree.numNodes(), true);
        m_d_ytsSizeWorkspace = std::make_unique<DTensor<T>>(m_data.nullDim(), 1, m_tree.numNonleafNodes(), true);
        m_d_adjDual = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_adjDualPrev = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_opPrim = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_opPrimPrev = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_primErr = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_dualErr = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        m_d_deltaPrim = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_deltaDual = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        /* Slice and reshape primal and dual workspaces */
        reshapePrimalWorkspace();
        reshapeDualWorkspace();
        /* Initialise projectable objects */
        initialiseProjectable();
    }

    ~Cache() = default;

    /**
     * Public methods
     */
    void cpIter();

    int cpAlgo(std::vector<T> &, std::vector<T> * = nullptr);

    int cpTime(std::vector<T> &);

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

    friend void testComputeErrors<>(CacheTestData<T> &d, T epsilon);
};

template<typename T>
void Cache<T>::initialiseSizes() {
    m_sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
    m_sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
    m_sizeY = m_tree.numNonleafNodes() * m_data.yDim();  ///< Y for all nonleaf nodes
    m_sizeT = m_tree.numNodes();  ///< T for all child nodes
    m_sizeS = m_tree.numNodes();  ///< S for all child nodes
    m_primSize = m_sizeU + m_sizeX + m_sizeY + m_sizeT + m_sizeS;
    m_sizeI = m_tree.numNonleafNodes() * m_data.yDim();
    m_sizeII = m_tree.numNonleafNodes();
    if (m_data.nonleafConstraint()[0]->isNone()){
        m_sizeIII = 0;
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
        m_sizeIII = m_tree.numNonleafNodes() * m_data.numStatesAndInputs();
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO */
    } else { constraintNotSupported(); }
    m_sizeIV = m_tree.numNodes() * (m_data.numStatesAndInputs() + 2);
    if (m_data.leafConstraint()[0]->isNone()){
        m_sizeV = 0;
    } else if (m_data.leafConstraint()[0]->isRectangle()) {
        m_sizeV = m_tree.numLeafNodes() * m_data.numStates();
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO */
    } else { constraintNotSupported(); }
    m_sizeVI = m_tree.numLeafNodes() * (m_data.numStates() + 2);
    m_dualSize = m_sizeI + m_sizeII + m_sizeIII + m_sizeIV + m_sizeV + m_sizeVI;
}

template<typename T>
void Cache<T>::reshapePrimalWorkspace() {
    size_t rowAxis = 0;
    size_t start = 0;
    m_d_u = std::make_unique<DTensor<T>>(*m_d_primWorkspace, rowAxis, start, start + m_sizeU - 1);
    m_d_u->reshape(m_data.numInputs(), 1, m_tree.numNonleafNodes());
    start += m_sizeU;
    m_d_x = std::make_unique<DTensor<T>>(*m_d_primWorkspace, rowAxis, start, start + m_sizeX - 1);
    m_d_x->reshape(m_data.numStates(), 1, m_tree.numNodes());
    start += m_sizeX;
    m_d_y = std::make_unique<DTensor<T>>(*m_d_primWorkspace, rowAxis, start, start + m_sizeY - 1);
    m_d_y->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    start += m_sizeY;
    m_d_t = std::make_unique<DTensor<T>>(*m_d_primWorkspace, rowAxis, start, start + m_sizeT - 1);
    m_d_t->reshape(1, 1, m_tree.numNodes());
    start += m_sizeT;
    m_d_s = std::make_unique<DTensor<T>>(*m_d_primWorkspace, rowAxis, start, start + m_sizeS - 1);
    m_d_s->reshape(1, 1, m_tree.numNodes());
}

template<typename T>
void Cache<T>::reshapeDualWorkspace() {
    size_t rowAxis = 0;
    size_t start = 0;
    m_d_i = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeI - 1);
    m_d_i->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    start += m_sizeI;
    m_d_ii = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeII - 1);
    m_d_ii->reshape(1, 1, m_tree.numNonleafNodes());
    start += m_sizeII;
    if (m_sizeIII) {
        m_d_iii = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeIII - 1);
        m_d_iii->reshape(m_data.numStatesAndInputs(), 1, m_tree.numNonleafNodes());
    }
    start += m_sizeIII;
    m_d_iv = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeIV - 1);
    m_d_iv->reshape(m_data.numStatesAndInputs() + 2, 1, m_tree.numNodes());
    /*
     * SocProjection requires one matrix, where the columns are the vectors.
     */
    m_d_ivSoc = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeIV - 1);
    m_d_ivSoc->reshape(m_data.numStatesAndInputs() + 2, m_tree.numNodes(), 1);
    start += m_sizeIV;
    if (m_sizeV) {
        m_d_v = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeV - 1);
        m_d_v->reshape(m_data.numStates(), 1, m_tree.numLeafNodes());
    }
    start += m_sizeV;
    m_d_vi = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeVI - 1);
    m_d_vi->reshape(m_data.numStates() + 2, 1, m_tree.numLeafNodes());
    /*
     * SocProjection requires one matrix, where the columns are the vectors.
     */
    m_d_viSoc = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, rowAxis, start, start + m_sizeVI - 1);
    m_d_viSoc->reshape(m_data.numStates() + 2, m_tree.numLeafNodes(), 1);
}

template<typename T>
void Cache<T>::initialiseProjectable() {
    /* I */
    m_cartRisk = std::make_unique<Cartesian<T>>();
    for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) { m_cartRisk->addCone(m_data.risk()[i]->cone()); }
    /* II */
    m_nnocNonleaf = std::make_unique<NonnegativeOrthantCone<T>>(m_tree.numNonleafNodes());
    /* III */
    if (m_data.nonleafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
            m_d_loBoundNonleaf = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs(), 1, m_tree.numNonleafNodes());
            m_d_hiBoundNonleaf = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs(), 1, m_tree.numNonleafNodes());
            for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
                DTensor<T> lo(*m_d_loBoundNonleaf, m_matAxis, i, i);
                DTensor<T> hi(*m_d_hiBoundNonleaf, m_matAxis, i, i);
                m_data.nonleafConstraint()[i]->lo().deviceCopyTo(lo);
                m_data.nonleafConstraint()[i]->hi().deviceCopyTo(hi);
            }
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO! */
    } else { constraintNotSupported(); }
    /* IV */
    m_socsNonleaf = std::make_unique<SocProjection<T>>(*m_d_ivSoc);
    m_d_socsNonleafHalves = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs() + 2, 1, m_tree.numNodes());
    std::vector<T> nonleafHalves(m_data.numStatesAndInputs() + 2, 0.);
    nonleafHalves[m_data.numStatesAndInputs()] = -.5;
    nonleafHalves[m_data.numStatesAndInputs() + 1] = .5;
    for (size_t i = 1; i < m_tree.numNodes(); i++) {
        DTensor<T> node(*m_d_socsNonleafHalves, m_matAxis, i, i);
        node.upload(nonleafHalves);
    }
    /* V */
    if (m_data.leafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.leafConstraint()[0]->isRectangle()) {
        m_d_loBoundLeaf = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numLeafNodes());
        m_d_hiBoundLeaf = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numLeafNodes());
        for (size_t i = 0; i < m_tree.numLeafNodes(); i++) {
            DTensor<T> lo(*m_d_loBoundLeaf, m_matAxis, i, i);
            DTensor<T> hi(*m_d_hiBoundLeaf, m_matAxis, i, i);
            m_data.leafConstraint()[i]->lo().deviceCopyTo(lo);
            m_data.leafConstraint()[i]->hi().deviceCopyTo(hi);
        }
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO! */
    } else { constraintNotSupported(); }
    /* VI */
    m_socsLeaf = std::make_unique<SocProjection<T>>(*m_d_viSoc);
    m_d_socsLeafHalves = std::make_unique<DTensor<T>>(m_data.numStates() + 2, 1, m_tree.numLeafNodes());
    std::vector<T> leafHalves(m_data.numStates() + 2, 0.);
    leafHalves[m_data.numStates()] = -.5;
    leafHalves[m_data.numStates() + 1] = .5;
    for (size_t i = 0; i < m_tree.numLeafNodes(); i++) {
        DTensor<T> node(*m_d_socsLeafHalves, m_matAxis, i, i);
        node.upload(leafHalves);
    }
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

/**
 * Proximal operator of `alpha * Id` on `s` at root node.
 * This operates on `primWorkspace`
 */
template<typename T>
void Cache<T>::proxRootS() {
    DTensor<T> sRoot(*m_d_s, m_matAxis, 0, 0);
    sRoot -= m_data.d_stepSize();
}

template<typename T>
void Cache<T>::projectPrimalWorkspaceOnDynamics() {
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
void Cache<T>::projectPrimalWorkspaceOnKernels() {
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

template<typename T>
void Cache<T>::translateSocs() {
    *m_d_iv += *m_d_socsNonleafHalves;
    *m_d_vi += *m_d_socsLeafHalves;
}

template<typename T>
void Cache<T>::projectDualWorkspaceOnConstraints() {
    /* I */
    m_cartRisk->projectOnDual(*m_d_i);
    /* II */
    m_nnocNonleaf->project(*m_d_ii);
    /* III */
    if (m_data.nonleafConstraint()[0]->isRectangle()) {
        size_t s = m_d_iii->numEl() - m_data.numStates();
        k_projectRectangle<<<numBlocks(s, TPB), TPB>>>(s, m_d_iii->raw() + m_data.numStates(),
                                                       m_d_loBoundNonleaf->raw() + m_data.numStates(),
                                                       m_d_hiBoundNonleaf->raw() + m_data.numStates());
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO!  */
    }
    /* IV */
    m_socsNonleaf->project(*m_d_ivSoc);
    /* V */
    if (m_data.leafConstraint()[0]->isRectangle()) {
        k_projectRectangle<<<numBlocks(m_d_v->numEl(), TPB), TPB>>>(m_d_v->numEl(), m_d_v->raw(),
                                                                    m_d_loBoundLeaf->raw(), m_d_hiBoundLeaf->raw());
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO!  */
    }
    /* VI */
    m_socsLeaf->project(*m_d_viSoc);
}

/**
 * Compute primalBar
 */
template<typename T>
void Cache<T>::modifyPrimal() {
    m_d_adjDual->deviceCopyTo(*m_d_primWorkspace);
    *m_d_primWorkspace *= -m_data.stepSize();
    *m_d_primWorkspace += *m_d_primPrev;
}

/**
 * Compute proximal of f on primalBar
 */
template<typename T>
void Cache<T>::proximalPrimal() {
    proxRootS();
    projectPrimalWorkspaceOnDynamics();
    projectPrimalWorkspaceOnKernels();
    m_d_primWorkspace->deviceCopyTo(*m_d_prim);  // Store new primal
}

/**
 * Compute dualBar
 */
template<typename T>
void Cache<T>::modifyDual() {
    m_L.op(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    m_d_opPrim->deviceCopyTo(*m_d_opPrimPrev);  // Save previous op of primal for error computation
    m_d_dualWorkspace->deviceCopyTo(*m_d_opPrim);  // Save op of primal for error computation
    *m_d_dualWorkspace *= 2.;
    *m_d_dualWorkspace -= *m_d_opPrimPrev;
    *m_d_dualWorkspace *= m_data.stepSize();
    *m_d_dualWorkspace += *m_d_dualPrev;
}

/**
 * Compute proximal of g* on dualBar
 */
template<typename T>
void Cache<T>::proximalDual() {
    *m_d_dualWorkspace *= m_data.stepSizeRecip();
    translateSocs();
    m_d_dualWorkspace->deviceCopyTo(*m_d_dual);
    projectDualWorkspaceOnConstraints();
    *m_d_dual -= *m_d_dualWorkspace;
    *m_d_dual *= m_data.stepSize();
}

/**
 * Compute adjoint of new dual
 */
template<typename T>
void Cache<T>::computeAdjDual() {
    m_d_dual->deviceCopyTo(*m_d_dualWorkspace);
    m_L.adj(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    m_d_adjDual->deviceCopyTo(*m_d_adjDualPrev);  // Save previous adjoint of dual for error computation
    m_d_primWorkspace->deviceCopyTo(*m_d_adjDual);  // Save adjoint of dual for error computation
}

template<typename T>
bool Cache<T>::computeError(size_t idx) {
    m_d_prim->download(m_cachePrim[idx]);
    m_d_dual->download(m_cacheDual[idx]);
    /* Primal error */
    m_d_primPrev->deviceCopyTo(*m_d_primWorkspace);
    *m_d_primWorkspace -= *m_d_prim;
    *m_d_primWorkspace *= m_data.stepSizeRecip();
    *m_d_primWorkspace -= *m_d_adjDualPrev;
    *m_d_primWorkspace += *m_d_adjDual;
    m_d_primWorkspace->deviceCopyTo(*m_d_primErr);
    /* Dual error */
    m_d_dualPrev->deviceCopyTo(*m_d_dualWorkspace);
    *m_d_dualWorkspace -= *m_d_dual;
    *m_d_dualWorkspace *= m_data.stepSizeRecip();
    *m_d_dualWorkspace -= *m_d_opPrimPrev;
    *m_d_dualWorkspace += *m_d_opPrim;
    m_d_dualWorkspace->deviceCopyTo(*m_d_dualErr);
    /* Inf-norm of errors */
    T primErr = m_d_primErr->maxAbs();
    T dualErr = m_d_dualErr->maxAbs();
    m_cacheError1[idx] = primErr;
    m_cacheError2[idx] = dualErr;
    /* Primal-dual error (avoid extra L adj until prim and dual errors pass tol) */
//    if (primErr <= m_tol && dualErr <= m_tol) {
    m_L.adj(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    *m_d_primWorkspace += *m_d_primErr;
    m_err = m_d_primWorkspace->maxAbs();
    m_cacheError0[idx] = m_err;
    if (m_err <= m_tol) { return true; }
//    } else {
//    m_cacheError0[idx] = max(primErr, dualErr);
//    }
    /* Infeasibility detection */
    if (m_detectInfeas) {
        /* Primal */
        m_d_prim->deviceCopyTo(*m_d_deltaPrim);
        *m_d_deltaPrim -= *m_d_primPrev;
        m_cacheDeltaPrim[idx] = m_d_deltaPrim->maxAbs();
        m_d_deltaPrim->deviceCopyTo(*m_d_primWorkspace);
        m_L.adj(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
        m_cacheNrmLtrDeltaDual[idx] = m_d_dualWorkspace->normF();
        /* Dual */
        m_d_dual->deviceCopyTo(*m_d_deltaDual);
        *m_d_deltaDual -= *m_d_dualPrev;
        m_cacheDeltaDual[idx] = m_d_deltaDual->maxAbs();
    }
    return false;
}

/**
 * Compute one (1) iteration of vanilla CP algorithm, nothing more.
 */
template<typename T>
void Cache<T>::cpIter() {
    m_d_prim->deviceCopyTo(*m_d_primPrev);  // Update previous primal
    m_d_dual->deviceCopyTo(*m_d_dualPrev);  // Update previous dual
    modifyPrimal();
    proximalPrimal();
    modifyDual();
    proximalDual();
    computeAdjDual();
}

/**
 * Add a vector to a .json file with a referenced name
 */
template<typename T>
static void
addArrayToJsonRef(rapidjson::Document &doc, rapidjson::GenericStringRef<char> const &name, std::vector<T> &vec) {
    rapidjson::Value array(rapidjson::kArrayType);
    for (size_t i = 0; i < vec.size(); i++) {
        array.PushBack(vec[i], doc.GetAllocator());
    }
    doc.AddMember(name, array, doc.GetAllocator());
}

/**
 * Add a vector to a .json file with a literal name
 */
template<typename T>
static void
addArrayToJsonStr(rapidjson::Document &doc, std::string const &name, std::vector<T> &vec) {
    rapidjson::Value array(rapidjson::kArrayType);
    for (size_t i = 0; i < vec.size(); i++) {
        array.PushBack(vec[i], doc.GetAllocator());
    }
    rapidjson::Value n;
    char buff[10];
    int len = sprintf(buff, "%s", name.c_str());
    n.SetString(buff, len, doc.GetAllocator());
    doc.AddMember(n, array, doc.GetAllocator());
}

/**
 * Print data to .json file
 */
template<typename T>
void Cache<T>::printToJson() {
    char text[1000000];
    rapidjson::MemoryPoolAllocator<> allocator(text, sizeof(text));
    rapidjson::Document doc(&allocator, 1024);
    doc.SetObject();
    doc.AddMember("maxIters", m_maxIters, doc.GetAllocator());
    doc.AddMember("tol", m_tol, doc.GetAllocator());
    doc.AddMember("sizeCache", m_maxIters - m_warmupIters, doc.GetAllocator());
    doc.AddMember("sizePrim", m_primSize, doc.GetAllocator());
    doc.AddMember("sizeDual", m_dualSize, doc.GetAllocator());
    rapidjson::GenericStringRef<char> nErr0 = "err0";
    addArrayToJsonRef(doc, nErr0, m_cacheError0);
    rapidjson::GenericStringRef<char> nErr1 = "err1";
    addArrayToJsonRef(doc, nErr1, m_cacheError1);
    rapidjson::GenericStringRef<char> nErr2 = "err2";
    addArrayToJsonRef(doc, nErr2, m_cacheError2);

    for (size_t i = 0; i < m_cachePrim.size(); i++) {
        std::string nPrim = "prim_" + std::to_string(i);
        addArrayToJsonStr(doc, nPrim, m_cachePrim[i]);
        std::string nDual = "dual_" + std::to_string(i);
        addArrayToJsonStr(doc, nDual, m_cacheDual[i]);
    }

    rapidjson::GenericStringRef<char> nDeltaPrim = "deltaPrim";
    addArrayToJsonRef(doc, nDeltaPrim, m_cacheDeltaPrim);
    rapidjson::GenericStringRef<char> nDeltaDual = "deltaDual";
    addArrayToJsonRef(doc, nDeltaDual, m_cacheDeltaDual);
    rapidjson::GenericStringRef<char> nNrmLtrDeltaDual = "nrmLtrDeltaDual";
    addArrayToJsonRef(doc, nNrmLtrDeltaDual, m_cacheNrmLtrDeltaDual);
    typedef rapidjson::GenericStringBuffer<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>> StringBuffer;
    StringBuffer buffer(&allocator);
    rapidjson::Writer<StringBuffer> writer(buffer, reinterpret_cast<rapidjson::CrtAllocator *>(&allocator));
    doc.Accept(writer);
    std::string json(buffer.GetString(), buffer.GetSize());
    std::ofstream of("/home/biggirl/Documents/remote_host/raocp-parallel/json/errorCache.json");
    of << json;
    if (!of.good()) throw std::runtime_error("[Cache::printToJson] Can't write the JSON string to the file!");
}

/**
 * Compute iterations of vanilla CP algorithm and check error.
 */
template<typename T>
int Cache<T>::cpAlgo(std::vector<T> &initState, std::vector<T> *previousSolution) {
    initialiseState(initState);
    /* Load previous solution if given */
    if (previousSolution) m_d_prim->upload(*previousSolution);
    /* Warm-up algorithm */
    for (size_t i = 0; i < m_warmupIters; i++) { cpIter(); }
    /* Run algorithm */
    size_t iters = m_maxIters - m_warmupIters;
    bool status = false;
    for (size_t i = 0; i < iters; i++) {
        /* Compute CP iteration */
        cpIter();
        /* Compute, store, and check error */
        status = computeError(i);
        if (status) {
            m_countIterations = m_warmupIters + i;
            break;
        }
    }
    /* Return status */
    if (status) {
        std::cout << "Converged in " << m_countIterations << " iterations, to a tolerance of " << m_tol << "\n";
        return 0;
    } else {
        std::cout << "Max iterations " << m_maxIters << " reached.\n";
        return 1;
    }
}

/**
 * Time vanilla CP algorithm with a parallelised cache
 */
template<typename T>
int Cache<T>::cpTime(std::vector<T> &initialState) {
    std::cout << "timer started" << "\n";
    const auto tick = std::chrono::high_resolution_clock::now();
    /* Run vanilla CP algorithm */
    int status = cpAlgo(initialState);
    const auto tock = std::chrono::high_resolution_clock::now();
    auto durationMilli = std::chrono::duration<double, std::milli>(tock - tick).count();
    std::cout << "timer stopped: " << durationMilli << " ms" << "\n";
    printToJson();
    return status;
}


#endif
