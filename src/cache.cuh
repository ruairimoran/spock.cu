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


TEMPLATE_WITH_TYPE_T
static void parseVec(const rapidjson::Value &value, std::vector<T> &vec) {
    size_t numElements = value.Capacity();
    if (vec.capacity() != numElements) vec.resize(numElements);
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        vec[i] = value[i].GetDouble();
    }
}

template<typename T>
__global__ void k_setToZero(T *, size_t);

template<typename T>
__global__ void k_shiftDiagonal(T *, T *, size_t, size_t = 0);

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

template<typename T>
void testDotM(CacheTestData<T> &, T);

template<typename T>
void testAndersonDirection(CacheTestData<T> &, T);


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
    bool m_status = false;
    bool m_detectInfeas = false;
    T m_tol = 0;
    T m_err = 0;
    size_t m_maxOuterIters = 0;
    size_t m_andSize = 0;
    size_t m_countIterations = 0;
    size_t m_rowAxis = 0;
    size_t m_colAxis = 1;
    size_t m_matAxis = 2;
    size_t m_period = 100;
    size_t m_callsToL = 0;
    /* Sizes */
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
    size_t m_primSize = 0;
    size_t m_dualSize = 0;
    size_t m_pdSize = 0;
    /* Iterates */
    std::unique_ptr<DTensor<T>> m_d_pd = nullptr;
    std::unique_ptr<DTensor<T>> m_d_prim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_pdPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_primPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ell = nullptr;
    std::unique_ptr<DTensor<T>> m_d_adjDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_opPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_adjDualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_opPrimPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_pdCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_primCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_adjDualCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_opPrimCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaIterate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaEll = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaResidual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_residual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_residualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_direction = nullptr;
    std::unique_ptr<DTensor<T>> m_d_scaledDirection = nullptr;
    /* Workspaces */
    std::unique_ptr<DTensor<T>> m_d_initState = nullptr;
    std::unique_ptr<DTensor<T>> m_d_pdWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_primWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_u = nullptr;
    std::unique_ptr<DTensor<T>> m_d_x = nullptr;
    std::unique_ptr<DTensor<T>> m_d_y = nullptr;
    std::unique_ptr<DTensor<T>> m_d_t = nullptr;
    std::unique_ptr<DTensor<T>> m_d_s = nullptr;
    std::unique_ptr<DTensor<T>> m_d_i = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iNnoc = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ii = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iii = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iv = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ivSoc = nullptr;
    std::unique_ptr<DTensor<T>> m_d_v = nullptr;
    std::unique_ptr<DTensor<T>> m_d_vi = nullptr;
    std::unique_ptr<DTensor<T>> m_d_viSoc = nullptr;
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_d = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ytsSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_pdDot = nullptr;
    std::unique_ptr<DTensor<T>> m_d_primDot = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualDot = nullptr;
    /* Projections */
    std::unique_ptr<SocProjection<T>> m_socsNonleaf = nullptr;
    std::unique_ptr<SocProjection<T>> m_socsLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_socsNonleafHalves = nullptr;
    std::unique_ptr<DTensor<T>> m_d_socsLeafHalves = nullptr;
    std::unique_ptr<NonnegativeOrthantCone<T>> m_nnocNonleaf = nullptr;
    std::unique_ptr<IndexedNnocProjection<T>> m_cartRiskDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_loBoundNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_hiBoundNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_loBoundLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_hiBoundLeaf = nullptr;
    /* Errors */
    std::unique_ptr<DTensor<T>> m_d_primErr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dualErr = nullptr;
    /* Caches */
    std::vector<size_t> m_cacheCallsToL;
    std::vector<T> m_cacheError0;
    std::vector<T> m_cacheError1;
    std::vector<T> m_cacheError2;
    std::vector<T> m_cacheDeltaPrim;
    std::vector<T> m_cacheDeltaDual;
    std::vector<T> m_cacheNrmLtrDeltaDual;
    std::vector<T> m_cacheDistDeltaDual;
    std::vector<T> m_cacheSuppDeltaDual;
    /* SuperMann */
    size_t m_maxInnerIters = 0;
    T m_c0 = 0.99;
    T m_c1 = 0.99;
    T m_c2 = 0.99;
    T m_beta = 0.5;
    T m_sigma = 0.1;
    T m_lambda = 1.0;
    /* Anderson */
    std::unique_ptr<DTensor<T>> m_d_andIterateMatrix = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andIterateMatrixLeft = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andIterateMatrixRight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andIterateMatrixCol0 = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andResidualMatrix = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andResidualMatrixLeft = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andResidualMatrixRight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andResidualMatrixCol0 = nullptr;
    std::unique_ptr<QRFactoriser<T>> m_andQRFactor = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andQR = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andQRGammaFull = nullptr;
    std::unique_ptr<DTensor<T>> m_d_andQRGamma = nullptr;

    /**
     * Protected methods
     */
    void reshape();

    void reshapePrimalWorkspace();

    void reshapeDualWorkspace();

    void initialiseProjectable();

    void initialiseSizes();

    void initialisePrev(std::vector<T> *);

    void initialiseState(std::vector<T> &initState);

    void proxRootS();

    void projectPrimalWorkspaceOnDynamics();

    void projectPrimalWorkspaceOnKernels();

    void translateSocs();

    void projectDualWorkspaceOnConstraints();

    void L(bool = false);

    void Ltr();

    T dotM(DTensor<T> &, DTensor<T> &);

    T normM(DTensor<T> &, DTensor<T> &);

    void modifyPrimal();

    void proximalPrimal();

    void modifyDual();

    void proximalDual();

    void cpIter();

    void saveIterate();

    void saveResidual();

    void acceptCandidate();

    void computeResidual();

    T computeResidualNorm();

    void computeDeltaIterate();

    void computeDeltaResidual();

    size_t updateQR(size_t);

    void updateDirection(size_t);

    bool computeError(size_t);

    bool infeasibilityDetection(size_t);

    void printToJson(std::string &);

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree<T> &tree, ProblemData<T> &data, bool detectInfeas = false,
          T tol = 1e-3, size_t maxOuterIters = 1000, size_t maxInnerIters = 8, size_t andBuff = 3) :
        m_tree(tree), m_data(data), m_detectInfeas(detectInfeas),
        m_tol(tol), m_maxOuterIters(maxOuterIters), m_maxInnerIters(maxInnerIters) {
        if (andBuff <= 32) {
            m_andSize = andBuff;
        } else {
            err << "[Cache] Anderson buffer size must be <= 32.\n";
            throw std::invalid_argument(err.str());
        }
        /* Sizes */
        initialiseSizes();
        /* Allocate memory on host */
        m_cacheCallsToL = std::vector<size_t>(m_maxOuterIters);
        m_cacheError0 = std::vector<T>(m_maxOuterIters);
        m_cacheError1 = std::vector<T>(m_maxOuterIters);
        m_cacheError2 = std::vector<T>(m_maxOuterIters);
        m_cacheDeltaPrim = std::vector<T>(m_maxOuterIters);
        m_cacheDeltaDual = std::vector<T>(m_maxOuterIters);
        m_cacheNrmLtrDeltaDual = std::vector<T>(m_maxOuterIters);
        m_cacheDistDeltaDual = std::vector<T>(m_maxOuterIters);
        m_cacheSuppDeltaDual = std::vector<T>(m_maxOuterIters);
        /* Allocate memory on device */
        m_d_pd = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_pdPrev = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_ell = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_ellPrev = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_pdCandidate = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_ellCandidate = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_deltaIterate = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_deltaEll = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_deltaResidual = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_residual = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_residualPrev = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_andIterateMatrix = std::make_unique<DTensor<T>>(m_pdSize, m_andSize, 1, true);
        m_d_andResidualMatrix = std::make_unique<DTensor<T>>(m_pdSize, m_andSize, 1, true);
        m_d_andQR = std::make_unique<DTensor<T>>(m_pdSize, m_andSize, 1, true);
        m_d_andQRGammaFull = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_direction = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_scaledDirection = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_pdWorkspace = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_pdDot = std::make_unique<DTensor<T>>(m_pdSize, 1, 1, true);
        m_d_initState = std::make_unique<DTensor<T>>(m_data.numStates(), 1, 1, true);
        m_d_q = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_xSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_uSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        m_d_xuSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs(), 1, m_tree.numNodes(), true);
        m_d_ytsSizeWorkspace = std::make_unique<DTensor<T>>(m_data.nullDim(), 1, m_tree.numNonleafNodes(), true);
        m_d_primErr = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
        m_d_dualErr = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
//        m_d_deltaPrim = std::make_unique<DTensor<T>>(m_primSize, 1, 1, true);
//        m_d_deltaDual = std::make_unique<DTensor<T>>(m_dualSize, 1, 1, true);
        /* Slice and reshape tensors */
        reshape();
        /* Initialise projectable objects */
        initialiseProjectable();
    }

    ~Cache() = default;

    /**
     * Public methods
     */
    int runCp(std::vector<T> &, std::vector<T> * = nullptr);

    int runSpock(std::vector<T> &, std::vector<T> * = nullptr);

    int cpTime(std::vector<T> &);

    int spTime(std::vector<T> &);

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

    friend void testComputeErrors<>(CacheTestData<T> &, T);

    friend void testDotM<>(CacheTestData<T> &, T);

    friend void testAndersonDirection<>(CacheTestData<T> &, T);
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
    if (m_data.nonleafConstraint()[0]->isNone()) {
        m_sizeIII = 0;
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
        m_sizeIII = m_tree.numNonleafNodes() * m_data.numStatesAndInputs();
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO */
    } else { constraintNotSupported(); }
    m_sizeIV = m_tree.numNodes() * (m_data.numStatesAndInputs() + 2);
    if (m_data.leafConstraint()[0]->isNone()) {
        m_sizeV = 0;
    } else if (m_data.leafConstraint()[0]->isRectangle()) {
        m_sizeV = m_tree.numLeafNodes() * m_data.numStates();
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO */
    } else { constraintNotSupported(); }
    m_sizeVI = m_tree.numLeafNodes() * (m_data.numStates() + 2);
    m_dualSize = m_sizeI + m_sizeII + m_sizeIII + m_sizeIV + m_sizeV + m_sizeVI;
    m_pdSize = m_primSize + m_dualSize;
}

template<typename T>
void Cache<T>::reshape() {
    m_d_prim = std::make_unique<DTensor<T>>(*m_d_pd, m_rowAxis, 0, m_primSize - 1);
    m_d_dual = std::make_unique<DTensor<T>>(*m_d_pd, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_primPrev = std::make_unique<DTensor<T>>(*m_d_pdPrev, m_rowAxis, 0, m_primSize - 1);
    m_d_dualPrev = std::make_unique<DTensor<T>>(*m_d_pdPrev, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_adjDual = std::make_unique<DTensor<T>>(*m_d_ell, m_rowAxis, 0, m_primSize - 1);
    m_d_opPrim = std::make_unique<DTensor<T>>(*m_d_ell, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_adjDualPrev = std::make_unique<DTensor<T>>(*m_d_ellPrev, m_rowAxis, 0, m_primSize - 1);
    m_d_opPrimPrev = std::make_unique<DTensor<T>>(*m_d_ellPrev, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_primCandidate = std::make_unique<DTensor<T>>(*m_d_pdCandidate, m_rowAxis, 0, m_primSize - 1);
    m_d_dualCandidate = std::make_unique<DTensor<T>>(*m_d_pdCandidate, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_adjDualCandidate = std::make_unique<DTensor<T>>(*m_d_ellCandidate, m_rowAxis, 0, m_primSize - 1);
    m_d_opPrimCandidate = std::make_unique<DTensor<T>>(*m_d_ellCandidate, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_primDot = std::make_unique<DTensor<T>>(*m_d_pdDot, m_rowAxis, 0, m_primSize - 1);
    m_d_dualDot = std::make_unique<DTensor<T>>(*m_d_pdDot, m_rowAxis, m_primSize, m_pdSize - 1);
    m_d_andIterateMatrixLeft = std::make_unique<DTensor<T>>(*m_d_andIterateMatrix, m_colAxis, 0, m_andSize - 2);
    m_d_andIterateMatrixRight = std::make_unique<DTensor<T>>(*m_d_andIterateMatrix, m_colAxis, 1, m_andSize - 1);
    m_d_andIterateMatrixCol0 = std::make_unique<DTensor<T>>(*m_d_andIterateMatrix, m_colAxis, 0, 0);
    m_d_andResidualMatrixLeft = std::make_unique<DTensor<T>>(*m_d_andResidualMatrix, m_colAxis, 0, m_andSize - 2);
    m_d_andResidualMatrixRight = std::make_unique<DTensor<T>>(*m_d_andResidualMatrix, m_colAxis, 1, m_andSize - 1);
    m_d_andResidualMatrixCol0 = std::make_unique<DTensor<T>>(*m_d_andResidualMatrix, m_colAxis, 0, 0);
    m_d_andQRGamma = std::make_unique<DTensor<T>>(*m_d_andQRGammaFull, m_rowAxis, 0, m_andSize - 1);
    reshapePrimalWorkspace();
    reshapeDualWorkspace();
}

template<typename T>
void Cache<T>::reshapePrimalWorkspace() {
    size_t rowAxis = 0;
    m_d_primWorkspace = std::make_unique<DTensor<T>>(*m_d_pdWorkspace, rowAxis, 0, m_primSize - 1);
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
    m_d_dualWorkspace = std::make_unique<DTensor<T>>(*m_d_pdWorkspace, m_rowAxis, m_primSize, m_pdSize - 1);
    size_t start = 0;
    m_d_i = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeI - 1);
    m_d_i->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    /*
     * IndexedNnocProjection requires [n x 1 x 1] tensor.
     */
    m_d_iNnoc = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeI - 1);
    start += m_sizeI;
    m_d_ii = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeII - 1);
    m_d_ii->reshape(1, 1, m_tree.numNonleafNodes());
    start += m_sizeII;
    if (m_sizeIII) {
        m_d_iii = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeIII - 1);
        m_d_iii->reshape(m_data.numStatesAndInputs(), 1, m_tree.numNonleafNodes());
    }
    start += m_sizeIII;
    m_d_iv = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeIV - 1);
    m_d_iv->reshape(m_data.numStatesAndInputs() + 2, 1, m_tree.numNodes());
    /*
     * SocProjection requires one matrix, where the columns are the vectors.
     */
    m_d_ivSoc = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeIV - 1);
    m_d_ivSoc->reshape(m_data.numStatesAndInputs() + 2, m_tree.numNodes(), 1);
    start += m_sizeIV;
    if (m_sizeV) {
        m_d_v = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeV - 1);
        m_d_v->reshape(m_data.numStates(), 1, m_tree.numLeafNodes());
    }
    start += m_sizeV;
    m_d_vi = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeVI - 1);
    m_d_vi->reshape(m_data.numStates() + 2, 1, m_tree.numLeafNodes());
    /*
     * SocProjection requires one matrix, where the columns are the vectors.
     */
    m_d_viSoc = std::make_unique<DTensor<T>>(*m_d_dualWorkspace, m_rowAxis, start, start + m_sizeVI - 1);
    m_d_viSoc->reshape(m_data.numStates() + 2, m_tree.numLeafNodes(), 1);
}

template<typename T>
void Cache<T>::initialiseProjectable() {
    /* I */
    m_cartRiskDual = std::make_unique<IndexedNnocProjection<T>>();
    for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) { m_cartRiskDual->addRisk(*(m_data.risk()[i])); }
    m_cartRiskDual->offline();
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
    /* QR */
    m_andQRFactor = std::make_unique<QRFactoriser<T>>(*m_d_andQR);
}

template<typename T>
void Cache<T>::initialiseState(std::vector<T> &initState) {
    /* Set initial state */
    if (initState.size() != m_data.numStates()) {
        err << "[initialiseState] Error initialising state: problem setup for " << m_data.numStates()
            << " but given " << initState.size() << " states" << "\n";
        throw std::invalid_argument(err.str());
    }
    m_d_initState->upload(initState);
}

/**
 * Initialise prev iterates.
 * - If given, load previous solution.
 * - Else, load ones.
 */
template<typename T>
void Cache<T>::initialisePrev(std::vector<T> *previousSolution) {
    if (previousSolution) {
        m_d_pd->upload(*previousSolution);
        saveIterate();
    } else {
        m_d_pdPrev->upload(std::vector<T>(m_pdSize, 1.));
        m_d_residualPrev->upload(std::vector<T>(m_pdSize, 1.));
    }
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
    DTensor<T> firstState(*m_d_x, m_matAxis, 0, 0);
    m_d_initState->deviceCopyTo(firstState);
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
    m_cartRiskDual->project(*m_d_iNnoc);
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
 * Call operator L on workspace
 */
template<typename T>
void Cache<T>::L(bool ignore) {
    m_L.op(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    if (!ignore) { m_callsToL += 1; }
}

/**
 * Call operator L' on workspace
 */
template<typename T>
void Cache<T>::Ltr() {
    m_L.adj(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
}

/**
 * Compute `x'My` where `M` is the preconditioning operator
 */
template<typename T>
T Cache<T>::dotM(DTensor<T> &x, DTensor<T> &y) {
    DTensor<T> yPrim(y, m_rowAxis, 0, m_primSize - 1);
    DTensor<T> yDual(y, m_rowAxis, m_primSize, m_pdSize - 1);
    yDual.deviceCopyTo(*m_d_dualWorkspace);
    Ltr();
    m_d_primWorkspace->deviceCopyTo(*m_d_primDot);
    yPrim.deviceCopyTo(*m_d_primWorkspace);
    L(true);
    m_d_dualWorkspace->deviceCopyTo(*m_d_dualDot);
    *m_d_pdDot *= -m_data.stepSize();
    *m_d_pdDot += y;
    return x.dotF(*m_d_pdDot);
}

/**
 * Compute `sqrt(x'My)` where `M` is the preconditioning operator
 */
template<typename T>
T Cache<T>::normM(DTensor<T> &x, DTensor<T> &y) {
    return sqrt(dotM(x, y));
}

/**
 * Compute primalBar
 */
template<typename T>
void Cache<T>::modifyPrimal() {
    m_d_dual->deviceCopyTo(*m_d_dualWorkspace);
    Ltr();
    m_d_primWorkspace->deviceCopyTo(*m_d_adjDualCandidate);  // Store adjoint of dual
    *m_d_primWorkspace *= -m_data.stepSize();
    *m_d_primWorkspace += *m_d_prim;
}

/**
 * Compute proximal of f on primalBar
 */
template<typename T>
void Cache<T>::proximalPrimal() {
    proxRootS();
    projectPrimalWorkspaceOnDynamics();
    projectPrimalWorkspaceOnKernels();
    m_d_primWorkspace->deviceCopyTo(*m_d_primCandidate);  // Store primal
}

/**
 * Compute dualBar
 */
template<typename T>
void Cache<T>::modifyDual() {
    L();
    m_d_dualWorkspace->deviceCopyTo(*m_d_opPrimCandidate);  // Store op of primal for error computation
    *m_d_dualWorkspace *= 2.;
    *m_d_dualWorkspace -= *m_d_opPrimPrev;
    *m_d_dualWorkspace *= m_data.stepSize();
    *m_d_dualWorkspace += *m_d_dual;
}

/**
 * Compute proximal of g* on dualBar
 */
template<typename T>
void Cache<T>::proximalDual() {
    *m_d_dualWorkspace *= m_data.stepSizeRecip();
    translateSocs();
    m_d_dualWorkspace->deviceCopyTo(*m_d_dualCandidate);
    projectDualWorkspaceOnConstraints();
    *m_d_dualCandidate -= *m_d_dualWorkspace;
    *m_d_dualCandidate *= m_data.stepSize();  // Store dual
}

/**
 * Compute one iteration of T(pd) operator, nothing more.
 * Write results to `candidates`.
 */
template<typename T>
void Cache<T>::cpIter() {
    modifyPrimal();
    proximalPrimal();
    modifyDual();
    proximalDual();
}

/**
 * Compute residual.
 */
template<typename T>
void Cache<T>::computeResidual() {
    m_d_pd->deviceCopyTo(*m_d_residual);
    *m_d_residual -= *m_d_pdCandidate;
}

/**
 * Compute residual norm.
 */
template<typename T>
T Cache<T>::computeResidualNorm() {
    computeResidual();
    return normM(*m_d_residual, *m_d_residual);
}

/**
 * Compute change of iterate.
 */
template<typename T>
void Cache<T>::computeDeltaIterate() {
    m_d_pd->deviceCopyTo(*m_d_deltaIterate);
    *m_d_deltaIterate -= *m_d_pdPrev;
    m_d_ell->deviceCopyTo(*m_d_deltaEll);
    *m_d_deltaEll -= *m_d_ellPrev;
}

/**
 * Compute change of residual.
 */
template<typename T>
void Cache<T>::computeDeltaResidual() {
//    m_d_residual->deviceCopyTo(*m_d_pdWorkspace);
//    std::cout << "Resid: " << m_d_x->tr();
//    m_d_residualPrev->deviceCopyTo(*m_d_pdWorkspace);
//    std::cout << "ResidPrev: " << m_d_x->tr();

    m_d_residual->deviceCopyTo(*m_d_deltaResidual);
    *m_d_deltaResidual -= *m_d_residualPrev;

//    m_d_deltaResidual->deviceCopyTo(*m_d_pdWorkspace);
//    std::cout << "deltaResid: " << m_d_x->tr();
}

/**
 * Save current iterates to prev.
 */
template<typename T>
void Cache<T>::saveIterate() {
    m_d_pd->deviceCopyTo(*m_d_pdPrev);
    m_d_ell->deviceCopyTo(*m_d_ellPrev);
}

/**
 * Save current residual to prev.
 */
template<typename T>
void Cache<T>::saveResidual() {
    m_d_residual->deviceCopyTo(*m_d_residualPrev);
}

/**
 * Save candidates as accepted iterates.
 */
template<typename T>
void Cache<T>::acceptCandidate() {
    m_d_pdCandidate->deviceCopyTo(*m_d_pd);
    m_d_ellCandidate->deviceCopyTo(*m_d_ell);
}

/**
 * Update QR decomposition.
 */
template<typename T>
size_t Cache<T>::updateQR(size_t idx) {
//    /**
//     * QR update to solve least-squares problem.
//     * 1. Shift Q right by 1.
//     * 2. Shift R right and down by 1.
//     * 3. Update first column of Q.
//     * 3. Modified Gram-Schmidt (for numerical stability).
//     * 4. Backward substitution.
//     */
//
//    /* Shift Q right by 1 */
//    m_d_andQLeft->deviceCopyTo(*m_d_andQRight);
//    /* Shift R right and down by 1 */
//    m_d_andRtr->deviceCopyTo(*m_d_andRtrCopy);
//    k_shiftDiagonal<<<1, dim3(32, 32)>>>(m_d_andRtr->raw(), m_d_andRtrCopy->raw(), m_andSize);
//
//    /* Store the new first column of m_d_andResidualMatrix as the first column of Q. */
//    m_d_andResidualMatrixCol0->deviceCopyTo(*m_d_andQCol0);
//    std::cout << "norm of new first column: " << m_d_andResidualMatrixCol0->normF() << "\n";
//
//    /* Modified Gram-Schmidt (orthogonalize against all columns j != 0 of Q, modified for numerical stability) */
//    std::vector<T> RtrCol0(m_andSize, 0.);
//    DTensor<T> d_one(std::vector<T>(1, 1.), 1);
//    size_t minSize = std::min(m_andSize, idx + 1);
//    T r;
//    if (idx != 0) {
//        for (size_t col = 1; col < minSize; col++) {
//            DTensor<T> QCol(*m_d_andQ, m_colAxis, col, col);
//            r = QCol.dotF(*m_d_andQCol0);
//            r *= 1 / QCol.normF();
//            /* Update first row of R */
//            RtrCol0[col] = r;
//            /* Update first column of Q */
//            m_d_andQCol0->addAB(QCol, d_one, -r, 1.);
//        }
//    }
//    /* Normalize first column of Q */
//    r = m_d_andQCol0->normF();
//    if (r) {
//        *m_d_andQCol0 *= 1 / r;
//    } else {
//        err << "[updateDirection] Attempt to divide by 0.\n";
//        throw std::invalid_argument(err.str());
//    }
//    /* Update first element of R and upload first row */
//    RtrCol0[0] = r;
//    m_d_andRtrCol0->upload(RtrCol0);
//    return minSize;
    return -1;
}

/**
 * Compute Anderson's direction.
 */
template<typename T>
void Cache<T>::updateDirection(size_t idx) {
    /* Shift iterate and residual matrices 1 column right */
    m_d_andIterateMatrixLeft->deviceCopyTo(*m_d_andIterateMatrixRight);
    m_d_andResidualMatrixLeft->deviceCopyTo(*m_d_andResidualMatrixRight);
    /* Update first column of iterate and residual matrices */
    m_d_deltaResidual->deviceCopyTo(*m_d_andResidualMatrixCol0);
    m_d_deltaIterate->deviceCopyTo(*m_d_andIterateMatrixCol0);
    *m_d_andIterateMatrixCol0 -= *m_d_deltaResidual;
//    m_d_andIterateMatrixCol0->deviceCopyTo(*m_d_pdWorkspace);
//    std::cout << "MP: " << m_d_x->tr();
//    m_d_andResidualMatrixCol0->deviceCopyTo(*m_d_pdWorkspace);
//    std::cout << "MR: " << m_d_x->tr();

    /**
     * 1. QR factorise `m_d_andResidualMatrix`.
     * 2. Compute least squares `m_d_andResidualMatrix \ m_d_residual`.
     * 4. Compute Anderson's direction.
     */
    if (idx >= m_andSize) {
        /* QR decomposition */
        m_d_andResidualMatrix->deviceCopyTo(*m_d_andQR);
        m_d_residual->deviceCopyTo(*m_d_andQRGammaFull);
        m_status = m_andQRFactor->factorise();
        if (m_status != 0) {
            err << "[updateDirection] QR factorisation returned status code: " << m_status << "\n";
            throw std::invalid_argument(err.str());
        }
        /* Least squares */
        m_status = m_andQRFactor->leastSquares(*m_d_andQRGammaFull);
        if (m_status != 0) {
            err << "[updateDirection] QR least squares returned status code: " << m_status << "\n";
            throw std::invalid_argument(err.str());
        }
        /* Compute new direction */
        m_d_direction->addAB(*m_d_andIterateMatrix, *m_d_andQRGamma, -1.);
        *m_d_direction -= *m_d_residual;
    } else {
        /* Use residual direction */
        m_d_residual->deviceCopyTo(*m_d_direction);
        *m_d_direction *= -1.;
    }
//    m_d_direction->deviceCopyTo(*m_d_pdWorkspace);
//    std::cout << "dir: " << m_d_x->tr();
}

/**
 * Compute errors for termination check.
 */
template<typename T>
bool Cache<T>::computeError(size_t idx) {
    cudaDeviceSynchronize();  // DO NOT REMOVE !!!
    /* -1/step*deltaIterate - L(deltaIterate) */
    m_d_deltaIterate->deviceCopyTo(*m_d_pdWorkspace);
    *m_d_pdWorkspace *= -m_data.stepSizeRecip();
    *m_d_pdWorkspace -= *m_d_deltaEll;
//    m_d_primWorkspace->deviceCopyTo(*m_d_primErr);
//    m_d_dualWorkspace->deviceCopyTo(*m_d_dualErr);
    /* Inf-norm of errors */
//    T primErr = m_d_primErr->maxAbs();
//    T dualErr = m_d_dualErr->maxAbs();
//    m_cacheError1[idx] = primErr;
//    m_cacheError2[idx] = dualErr;
    /* Primal-dual error (avoid extra adj until prim and dual errors pass relaxed tol) */
//    T relaxTol = m_tol * 10;
//    if (primErr <= relaxTol && dualErr <= relaxTol) {
//    Ltr();
//    *m_d_primWorkspace += *m_d_primErr;
    m_err = m_d_pdWorkspace->maxAbs();
//    m_err = std::max(primErr, dualErr);
    m_cacheError0[idx] = m_err;
    m_cacheCallsToL[idx] = m_callsToL;
    if (m_err <= m_tol) { return true; }
//    } else {
//        m_cacheError0[idx] = max(primErr, dualErr);
//    }
    return false;
}

/**
 * Infeasibility detection.
 */
template<typename T>
bool Cache<T>::infeasibilityDetection(size_t idx) {
//    /* Primal */
//    m_d_prim->deviceCopyTo(*m_d_deltaPrim);
//    *m_d_deltaPrim -= *m_d_primPrev;
//    m_cacheDeltaPrim[idx] = m_d_deltaPrim->maxAbs();
//    m_d_deltaPrim->deviceCopyTo(*m_d_primWorkspace);
//    Ltr();
//    m_cacheNrmLtrDeltaDual[idx] = m_d_dualWorkspace->normF();
//    /* Dual */
//    m_d_dual->deviceCopyTo(*m_d_deltaDual);
//    *m_d_deltaDual -= *m_d_dualPrev;
//    m_cacheDeltaDual[idx] = m_d_deltaDual->maxAbs();
    return false;
}

/**
 * CP algorithm.
 */
template<typename T>
int Cache<T>::runCp(std::vector<T> &initState, std::vector<T> *previousSolution) {
    initialiseState(initState);
    /* Load previous solution if given */
    initialisePrev(previousSolution);
    /* Run algorithm */
    for (size_t i = 0; i < m_maxOuterIters; i++) {
        if (i % m_period == 0) { std::cout << "." << std::flush; }
        /* Compute CP iteration */
        cpIter();
        /* Compute residual */
        computeResidual();
        /* Save candidate to accepted iterate */
        acceptCandidate();
        /* Compute change in iterate */
        computeDeltaIterate();
        /* Check error */
        m_status = computeError(i);
        if (m_status) {
            m_countIterations = i;
            break;
        }
        /* Save iterate to prev */
        saveIterate();
    }
    /* Return status */
    if (m_status) {
        std::cout << "\nConverged in " << m_countIterations << " iterations, to a tolerance of " << m_tol << "\n";
        return 0;
    } else {
        std::cout << "\nMax iterations (" << m_maxOuterIters << ") reached.\n";
        return 1;
    }
}

/**
 * SPOCK algorithm.
 */
template<typename T>
int Cache<T>::runSpock(std::vector<T> &initState, std::vector<T> *previousSolution) {
    initialiseState(initState);
    /* Load previous solution if given */
    initialisePrev(previousSolution);

    /* Initialise */
    size_t countK0 = 0;
    size_t countK1 = 0;
    size_t countK2 = 0;
    size_t countK3 = 0;
    T zeta = INFINITY;
    T w = INFINITY;
    T wSafe = INFINITY;
    T wTilde = INFINITY;
    T rho = 0;
    T tau = 0;
//    cpIter();
//    zeta = computeResidualNorm();
//    wSafe = zeta;
    for (size_t iOut = 0; iOut < m_maxOuterIters; iOut++) {
        if (iOut % m_period == 0) { std::cout << "." << std::flush; }
        if (iOut != 0) {
            /* Accept new iterate */
            acceptCandidate();
            /* Compute change of iterate */
            computeDeltaIterate();
            /* Check error */
            m_status = computeError(iOut);
            if (m_status) {
                m_countIterations = iOut;
                break;
            }
        }
        /* START: compute T */
        cpIter();
        /* Compute residual */
        computeResidual();
        /* Compute residual norm */
        w = normM(*m_d_residual, *m_d_residual);
        /* Compute change of residual */
        computeDeltaResidual();
        /* Save iterate to prev */
        saveIterate();
        /* Save residual to prev */
        saveResidual();
        /* Compute direction */
        updateDirection(iOut);
        /* Blind update */
        if (w <= m_c0 * zeta && false) {
            m_d_pdPrev->deviceCopyTo(*m_d_pdCandidate);
            *m_d_pdCandidate += *m_d_direction;
            zeta = w;
            countK0 += 1;
            continue;  // K0
        }
        /* Line search on tau */
        tau = 1.;
        for (size_t iIn = 0; iIn < m_maxInnerIters; iIn++) {
            m_d_pdPrev->deviceCopyTo(*m_d_pd);
            m_d_direction->deviceCopyTo(*m_d_scaledDirection);
            *m_d_scaledDirection *= tau;
            *m_d_pd += *m_d_scaledDirection;
            cpIter();
            wTilde = computeResidualNorm();
            /* Educated update */
            if (w <= wSafe && wTilde <= m_c1 * w) {
                m_d_pd->deviceCopyTo(*m_d_pdCandidate);
                wSafe = wTilde + pow(m_c2, iOut);
                countK1 += 1;
                break;  // K1
            }
//            rho = pow(wTilde, 2) - 2 * m_data.stepSize() * dotM(*m_d_residual, *m_d_scaledDirection);
            *m_d_scaledDirection *= -1.;
            *m_d_scaledDirection += *m_d_residual;
            rho = dotM(*m_d_residual, *m_d_scaledDirection);
            /* Safeguard update */
            if (rho >= m_sigma * wTilde * w) {
                *m_d_residual *= (m_lambda * rho / pow(wTilde, 2));
                m_d_pdPrev->deviceCopyTo(*m_d_pdCandidate);
                *m_d_pdCandidate -= *m_d_residual;
                countK2 += 1;
                break;  // K2
            } else {
                tau *= m_beta;
            }
            if (iIn >= m_maxInnerIters - 1) { countK3 += 1; }
        }
    }
    /* Return status */
    if (m_status) {
        std::cout << "\nConverged in " << m_countIterations << " outer iterations, to a tolerance of " << m_tol
                  << ", [K0: " << countK0
                  << ", K1: " << countK1
                  << ", K2: " << countK2
                  << ", K3: " << countK3
                  << "].\n";
        return 0;
    } else {
        std::cout << "\nMax iterations (" << m_maxOuterIters << ") reached [K0: " << countK0
                  << ", K1: " << countK1
                  << ", K2: " << countK2
                  << ", K3: " << countK3
                  << "].\n";
        return 1;
    }
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
void Cache<T>::printToJson(std::string &file) {
    char text[1000000];
    rapidjson::MemoryPoolAllocator<> allocator(text, sizeof(text));
    rapidjson::Document doc(&allocator, 2048);
    doc.SetObject();
    doc.AddMember("maxIters", m_maxOuterIters, doc.GetAllocator());
    doc.AddMember("tol", m_tol, doc.GetAllocator());
    doc.AddMember("sizeCache", m_maxOuterIters, doc.GetAllocator());
    doc.AddMember("sizePrim", m_primSize, doc.GetAllocator());
    doc.AddMember("sizeDual", m_dualSize, doc.GetAllocator());
//    std::vector<T> solution(m_primSize);
//    m_d_prim->download(solution);
//    rapidjson::GenericStringRef<char> nSol = "sol";
//    addArrayToJsonRef(doc, nSol, solution);
    rapidjson::GenericStringRef<char> nCallsL = "callsL";
    addArrayToJsonRef(doc, nCallsL, m_cacheCallsToL);
    rapidjson::GenericStringRef<char> nErr0 = "err0";
    addArrayToJsonRef(doc, nErr0, m_cacheError0);
    rapidjson::GenericStringRef<char> nErr1 = "err1";
    addArrayToJsonRef(doc, nErr1, m_cacheError1);
    rapidjson::GenericStringRef<char> nErr2 = "err2";
    addArrayToJsonRef(doc, nErr2, m_cacheError2);
//    rapidjson::GenericStringRef<char> nDeltaPrim = "deltaPrim";
//    addArrayToJsonRef(doc, nDeltaPrim, m_cacheDeltaPrim);
//    rapidjson::GenericStringRef<char> nDeltaDual = "deltaDual";
//    addArrayToJsonRef(doc, nDeltaDual, m_cacheDeltaDual);
//    rapidjson::GenericStringRef<char> nNrmLtrDeltaDual = "nrmLtrDeltaDual";
//    addArrayToJsonRef(doc, nNrmLtrDeltaDual, m_cacheNrmLtrDeltaDual);
    typedef rapidjson::GenericStringBuffer<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>> StringBuffer;
    StringBuffer buffer(&allocator);
    rapidjson::Writer<StringBuffer> writer(buffer, reinterpret_cast<rapidjson::CrtAllocator *>(&allocator));
    doc.Accept(writer);
    std::string json(buffer.GetString(), buffer.GetSize());
    std::ofstream of("/home/biggirl/Documents/remote_host/raocp-parallel/json/cache" + file + ".json");
    of << json;
    if (!of.good()) throw std::runtime_error("[Cache::printToJson] Can't write the JSON string to the file!");
}

/**
 * Time vanilla CP algorithm with a parallelised cache
 */
template<typename T>
int Cache<T>::cpTime(std::vector<T> &initialState) {
    std::cout << "cp timer started" << "\n";
    const auto tick = std::chrono::high_resolution_clock::now();
    /* Run vanilla CP algorithm */
    int status = runCp(initialState);
    const auto tock = std::chrono::high_resolution_clock::now();
    auto durationMilli = std::chrono::duration<double, std::milli>(tock - tick).count();
    std::cout << "cp timer stopped: " << durationMilli << " ms" << "\n";
    std::string n = "Cp";
    printToJson(n);
    return status;
}

/**
 * Time SPOCK algorithm with a parallelised cache
 */
template<typename T>
int Cache<T>::spTime(std::vector<T> &initialState) {
    std::cout << "spock timer started" << "\n";
    const auto tick = std::chrono::high_resolution_clock::now();
    /* Run SPOCK algorithm */
    int status = runSpock(initialState);
    const auto tock = std::chrono::high_resolution_clock::now();
    auto durationMilli = std::chrono::duration<double, std::milli>(tock - tick).count();
    std::cout << "spock timer stopped: " << durationMilli << " ms" << "\n";
    std::string n = "Sp";
    printToJson(n);
    return status;
}


#endif
