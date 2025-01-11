#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"
#include "risks.cuh"
#include "operator.cuh"
#include "projections.cuh"
#include <chrono>


TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyInTS(T *, T *, size_t, size_t, size_t, size_t *, size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyOutTS(T *, T *, size_t, size_t, size_t, size_t *, size_t *);

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
void testDotM(CacheTestData<T> &, T);


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
    T m_tolAbs = 0;
    T m_tolRel = 0;
    T m_tol = 0;
    T m_errAbs = 0;
    size_t m_maxOuterIters = 0;
    size_t m_andSize = 0;
    size_t m_countIterations = 0;
    size_t m_rowAxis = 0;
    size_t m_colAxis = 1;
    size_t m_matAxis = 2;
    size_t m_callsToL = 0;
    bool m_allowK0 = false;
    bool m_debug = false;
    bool m_errInit = false;
    bool m_status = false;
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
    size_t m_sizeIterate = 0;
    size_t m_sizePrim = 0;
    size_t m_sizeDual = 0;
    /* Iterates */
    std::unique_ptr<DTensor<T>> m_d_iterate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iteratePrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iterateDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iteratePrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iteratePrevPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iteratePrevDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iterateCandidate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iterateCandidatePrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iterateCandidateDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_residual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_residualPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_residualDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_residualPrev = nullptr;
    std::unique_ptr<DTensor<T>> m_d_direction = nullptr;
    std::unique_ptr<DTensor<T>> m_d_directionScaled = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaIterate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaIteratePrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaIterateDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_deltaResidual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellDeltaIterate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellDeltaIteratePrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellDeltaIterateDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_iterateBackup = nullptr;
    std::unique_ptr<DTensor<T>> m_d_err = nullptr;
    /* Workspaces */
    std::unique_ptr<DTensor<T>> m_d_initState = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workIterate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workIteratePrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workIterateDual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workX = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workU = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workXU = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workYTS = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workDot = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workDotPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workDotDual = nullptr;
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
    /* Projections */
    std::unique_ptr<SocProjection<T>> m_socsNonleaf = nullptr;
    std::unique_ptr<SocProjection<T>> m_socsLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_socsNonleafHalves = nullptr;
    std::unique_ptr<DTensor<T>> m_d_socsLeafHalves = nullptr;
    std::unique_ptr<NonnegativeOrthantCone<T>> m_nnocNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_loBoundNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_hiBoundNonleaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_loBoundLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_hiBoundLeaf = nullptr;
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
    T m_c0 = .99;
    T m_c1 = .99;
    T m_c2 = .99;
    T m_beta = .5;
    T m_sigma = .1;
    T m_lambda = 1.;
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

    void backup();

    void restore();

    void saveToPrev();

    void acceptCandidate();

    void computeResidual();

    void computeDeltaIterate();

    void computeDeltaResidual();

    void updateDirection(size_t);

    bool computeError(size_t);

    void printToJson(std::string &);

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree<T> &tree,
          ProblemData<T> &data,
          T absTol = 1e-3,
          T relTol = 0.,
          size_t maxOuterIters = 1000,
          size_t maxInnerIters = 8,
          size_t andBuff = 3,
          bool allowK0 = false,
          bool debug = false) :
        m_tree(tree), m_data(data), m_tolAbs(absTol), m_tolRel(relTol), m_maxOuterIters(maxOuterIters),
        m_maxInnerIters(maxInnerIters), m_andSize(andBuff), m_allowK0(allowK0), m_debug(debug) {
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
        m_d_initState = std::make_unique<DTensor<T>>(m_data.numStates(), 1, 1, true);
        m_d_iterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_iteratePrev = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_iterateCandidate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_deltaIterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_ellDeltaIterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_deltaResidual = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_residual = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_residualPrev = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_iterateBackup = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_err = std::make_unique<DTensor<T>>(m_sizePrim, 1, 1, true);
        m_d_andIterateMatrix = std::make_unique<DTensor<T>>(m_sizeIterate, m_andSize, 1, true);
        m_d_andResidualMatrix = std::make_unique<DTensor<T>>(m_sizeIterate, m_andSize, 1, true);
        m_d_andQR = std::make_unique<DTensor<T>>(m_sizeIterate, m_andSize, 1, true);
        m_d_andQRGammaFull = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_direction = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_directionScaled = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_workIterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_workX = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_workU = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        m_d_workXU = std::make_unique<DTensor<T>>(m_data.numStatesAndInputs(), 1, m_tree.numNodes(), true);
        m_d_workYTS = std::make_unique<DTensor<T>>(m_data.nullDim(), 1, m_tree.numNonleafNodes(), true);
        m_d_workDot = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_q = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        /* Slice and reshape tensors */
        reshape();
        /* Initialise projectable objects */
        initialiseProjectable();
    }

    ~Cache() = default;

    /**
     * Public methods
     */
    void reset();  // For testing.

    int runCp(std::vector<T> &, std::vector<T> * = nullptr);

    int runSpock(std::vector<T> &, std::vector<T> * = nullptr);

    int timeCp(std::vector<T> &);

    T timeSp(std::vector<T> &);

    /**
     * Getters
     */
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

    friend void testDotM<>(CacheTestData<T> &, T);
};

template<typename T>
void Cache<T>::reset() {
    m_L.reset();
    /* Create zero vectors */
    std::vector<T> numStates(m_data.numStates(), 0);
    std::vector<T> sizeIterate(m_sizeIterate, 0);
    std::vector<T> sizePrim(m_sizePrim, 0);
    std::vector<T> sizeIterateSizeAnd(m_sizeIterate * m_andSize, 0);
    std::vector<T> numStatesNumNodes(m_data.numStates() * m_tree.numNodes(), 0);
    std::vector<T> numInputsNumNodes(m_data.numInputs() * m_tree.numNodes(), 0);
    std::vector<T> numStatesAndInputsNumNodes(m_data.numStatesAndInputs() * m_tree.numNodes(), 0);
    std::vector<T> nullDimNumNonleafNodes(m_data.nullDim() * m_tree.numNonleafNodes(), 0);
    std::vector<T> numInputsNumNonleafNodes(m_data.numInputs() * m_tree.numNonleafNodes(), 0);
    /* Zero all cached data */
    std::fill(m_cacheCallsToL.begin(), m_cacheCallsToL.end(), 0);
    std::fill(m_cacheError0.begin(), m_cacheError0.end(), 0);
    std::fill(m_cacheError1.begin(), m_cacheError1.end(), 0);
    std::fill(m_cacheError2.begin(), m_cacheError2.end(), 0);
    std::fill(m_cacheDeltaPrim.begin(), m_cacheDeltaPrim.end(), 0);
    std::fill(m_cacheDeltaDual.begin(), m_cacheDeltaDual.end(), 0);
    std::fill(m_cacheNrmLtrDeltaDual.begin(), m_cacheNrmLtrDeltaDual.end(), 0);
    std::fill(m_cacheDistDeltaDual.begin(), m_cacheDistDeltaDual.end(), 0);
    std::fill(m_cacheSuppDeltaDual.begin(), m_cacheSuppDeltaDual.end(), 0);
    m_d_initState->upload(numStates);
    m_d_iterate->upload(sizeIterate);
    m_d_iteratePrev->upload(sizeIterate);
    m_d_iterateCandidate->upload(sizeIterate);
    m_d_deltaIterate->upload(sizeIterate);
    m_d_ellDeltaIterate->upload(sizeIterate);
    m_d_deltaResidual->upload(sizeIterate);
    m_d_residual->upload(sizeIterate);
    m_d_residualPrev->upload(sizeIterate);
    m_d_iterateBackup->upload(sizeIterate);
    m_d_err->upload(sizePrim);
    m_d_andIterateMatrix->upload(sizeIterateSizeAnd);
    m_d_andResidualMatrix->upload(sizeIterateSizeAnd);
    m_d_andQR->upload(sizeIterateSizeAnd);
    m_d_andQRGammaFull->upload(sizeIterate);
    m_d_direction->upload(sizeIterate);
    m_d_directionScaled->upload(sizeIterate);
    m_d_workIterate->upload(sizeIterate);
    m_d_workX->upload(numStatesNumNodes);
    m_d_workU->upload(numInputsNumNodes);
    m_d_workXU->upload(numStatesAndInputsNumNodes);
    m_d_workYTS->upload(nullDimNumNonleafNodes);
    m_d_workDot->upload(sizeIterate);
    m_d_q->upload(numStatesNumNodes);
    m_d_d->upload(numInputsNumNonleafNodes);
}

template<typename T>
void Cache<T>::initialiseSizes() {
    m_sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
    m_sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
    m_sizeY = m_tree.numNonleafNodes() * m_data.yDim();  ///< Y for all nonleaf nodes
    m_sizeT = m_tree.numNodes();  ///< T for all child nodes
    m_sizeS = m_tree.numNodes();  ///< S for all child nodes
    m_sizePrim = m_sizeU + m_sizeX + m_sizeY + m_sizeT + m_sizeS;
    m_sizeI = m_tree.numNonleafNodes() * m_data.yDim();
    m_sizeII = m_tree.numNonleafNodes();
    m_sizeIII = m_data.nonleafConstraint()->dimension();
    m_sizeIV = m_tree.numNodes() * (m_data.numStatesAndInputs() + 2);
    m_sizeV = m_data.leafConstraint()->dimension();
    m_sizeVI = m_tree.numLeafNodes() * (m_data.numStates() + 2);
    m_sizeDual = m_sizeI + m_sizeII + m_sizeIII + m_sizeIV + m_sizeV + m_sizeVI;
    m_sizeIterate = m_sizePrim + m_sizeDual;
}

template<typename T>
void Cache<T>::reshape() {
    m_d_iteratePrim = std::make_unique<DTensor<T>>(*m_d_iterate, m_rowAxis, 0, m_sizePrim - 1);
    m_d_iterateDual = std::make_unique<DTensor<T>>(*m_d_iterate, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    m_d_iteratePrevPrim = std::make_unique<DTensor<T>>(*m_d_iteratePrev, m_rowAxis, 0, m_sizePrim - 1);
    m_d_iteratePrevDual = std::make_unique<DTensor<T>>(*m_d_iteratePrev, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    m_d_iterateCandidatePrim = std::make_unique<DTensor<T>>(*m_d_iterateCandidate, m_rowAxis, 0, m_sizePrim - 1);
    m_d_iterateCandidateDual = std::make_unique<DTensor<T>>(*m_d_iterateCandidate, m_rowAxis, m_sizePrim,
                                                            m_sizeIterate - 1);
    m_d_residualPrim = std::make_unique<DTensor<T>>(*m_d_residual, m_rowAxis, 0, m_sizePrim - 1);
    m_d_residualDual = std::make_unique<DTensor<T>>(*m_d_residual, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    m_d_workDotPrim = std::make_unique<DTensor<T>>(*m_d_workDot, m_rowAxis, 0, m_sizePrim - 1);
    m_d_workDotDual = std::make_unique<DTensor<T>>(*m_d_workDot, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    m_d_deltaIteratePrim = std::make_unique<DTensor<T>>(*m_d_deltaIterate, m_rowAxis, 0, m_sizePrim - 1);
    m_d_deltaIterateDual = std::make_unique<DTensor<T>>(*m_d_deltaIterate, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    m_d_ellDeltaIteratePrim = std::make_unique<DTensor<T>>(*m_d_ellDeltaIterate, m_rowAxis, 0, m_sizePrim - 1);
    m_d_ellDeltaIterateDual = std::make_unique<DTensor<T>>(*m_d_ellDeltaIterate, m_rowAxis, m_sizePrim,
                                                           m_sizeIterate - 1);
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
    m_d_workIteratePrim = std::make_unique<DTensor<T>>(*m_d_workIterate, rowAxis, 0, m_sizePrim - 1);
    size_t start = 0;
    m_d_u = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeU - 1);
    m_d_u->reshape(m_data.numInputs(), 1, m_tree.numNonleafNodes());
    start += m_sizeU;
    m_d_x = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeX - 1);
    m_d_x->reshape(m_data.numStates(), 1, m_tree.numNodes());
    start += m_sizeX;
    m_d_y = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeY - 1);
    m_d_y->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    start += m_sizeY;
    m_d_t = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeT - 1);
    m_d_t->reshape(1, 1, m_tree.numNodes());
    start += m_sizeT;
    m_d_s = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeS - 1);
    m_d_s->reshape(1, 1, m_tree.numNodes());
}

template<typename T>
void Cache<T>::reshapeDualWorkspace() {
    m_d_workIterateDual = std::make_unique<DTensor<T>>(*m_d_workIterate, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    size_t start = 0;
    m_d_i = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeI - 1);
    m_d_i->reshape(m_data.yDim(), 1, m_tree.numNonleafNodes());
    /*
     * IndexedNnocProjection requires [n x 1 x 1] tensor.
     */
    m_d_iNnoc = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeI - 1);
    start += m_sizeI;
    m_d_ii = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeII - 1);
    m_d_ii->reshape(1, 1, m_tree.numNonleafNodes());
    start += m_sizeII;
    if (m_sizeIII) {
        m_d_iii = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeIII - 1);
        m_d_iii->reshape(m_data.numStatesAndInputs(), 1, m_tree.numNonleafNodes());
    }
    start += m_sizeIII;
    m_d_iv = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeIV - 1);
    m_d_iv->reshape(m_data.numStatesAndInputs() + 2, 1, m_tree.numNodes());
    /*
     * SocProjection requires one matrix, where the columns are the vectors.
     */
    m_d_ivSoc = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeIV - 1);
    m_d_ivSoc->reshape(m_data.numStatesAndInputs() + 2, m_tree.numNodes(), 1);
    start += m_sizeIV;
    if (m_sizeV) {
        m_d_v = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeV - 1);
        m_d_v->reshape(m_data.numStates(), 1, m_tree.numLeafNodes());
    }
    start += m_sizeV;
    m_d_vi = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeVI - 1);
    m_d_vi->reshape(m_data.numStates() + 2, 1, m_tree.numLeafNodes());
    /*
     * SocProjection requires one matrix, where the columns are the vectors.
     */
    m_d_viSoc = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeVI - 1);
    m_d_viSoc->reshape(m_data.numStates() + 2, m_tree.numLeafNodes(), 1);
}

template<typename T>
void Cache<T>::initialiseProjectable() {
    /* II */
    m_nnocNonleaf = std::make_unique<NonnegativeOrthantCone<T>>(m_tree.numNonleafNodes());
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
        m_d_iterate->upload(*previousSolution);
        saveToPrev();
    } else {
        m_d_iteratePrev->upload(std::vector<T>(m_sizeIterate, 1.));
        m_d_residualPrev->upload(std::vector<T>(m_sizeIterate, 1.));
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

}

template<typename T>
void Cache<T>::projectPrimalWorkspaceOnKernels() {
    /**
     * Project on kernel of every node of tree at once
     */
    /* Gather vec[i] = (y_i, t[ch(i)], s[ch(i)]) for all nonleaf nodes */
    memCpyNode2Node(*m_d_workYTS, *m_d_y, 0, m_tree.numNonleafNodesMinus1(), m_data.yDim());
    k_memCpyInTS<<<numBlocks(m_tree.numNodes(), TPB), TPB>>>(m_d_workYTS->raw(), m_d_t->raw(),
                                                             m_tree.numNodes(), m_d_workYTS->numRows(),
                                                             m_data.yDim(),
                                                             m_tree.d_ancestors().raw(), m_tree.d_childFrom().raw());
    k_memCpyInTS<<<numBlocks(m_tree.numNodes(), TPB), TPB>>>(m_d_workYTS->raw(), m_d_s->raw(),
                                                             m_tree.numNodes(), m_d_workYTS->numRows(),
                                                             m_data.yDim() + m_tree.numEvents(),
                                                             m_tree.d_ancestors().raw(), m_tree.d_childFrom().raw());
    /* Project onto nullspace in place */
    m_d_workYTS->addAB(m_data.risk()->nullspaceProj(), *m_d_workYTS);
    /* Disperse vec[i] = (y_i, t[ch(i)], s[ch(i)]) for all nonleaf nodes */
    memCpyNode2Node(*m_d_y, *m_d_workYTS, 0, m_tree.numNonleafNodesMinus1(), m_data.yDim());
    k_memCpyOutTS<<<numBlocks(m_tree.numNodes(), TPB), TPB>>>(m_d_t->raw(), m_d_workYTS->raw(),
                                                              m_tree.numNodes(), m_d_workYTS->numRows(),
                                                              m_data.yDim(),
                                                              m_tree.d_ancestors().raw(), m_tree.d_childFrom().raw());
    k_memCpyOutTS<<<numBlocks(m_tree.numNodes(), TPB), TPB>>>(m_d_s->raw(), m_d_workYTS->raw(),
                                                              m_tree.numNodes(), m_d_workYTS->numRows(),
                                                              m_data.yDim() + m_tree.numEvents(),
                                                              m_tree.d_ancestors().raw(), m_tree.d_childFrom().raw());
}

template<typename T>
void Cache<T>::translateSocs() {
    *m_d_iv += *m_d_socsNonleafHalves;
    *m_d_vi += *m_d_socsLeafHalves;
}

template<typename T>
void Cache<T>::projectDualWorkspaceOnConstraints() {
    /* I */
    m_data.risk()->projectDual(*m_d_iNnoc);
    /* II */
    m_nnocNonleaf->project(*m_d_ii);
    /* III */
    m_data.nonleafConstraint()->constrain(*m_d_iii);
    /* IV */
    m_socsNonleaf->project(*m_d_ivSoc);
    /* V */
    m_data.leafConstraint()->constrain(*m_d_v);
    /* VI */
    m_socsLeaf->project(*m_d_viSoc);
}

/**
 * Call operator L on workspace
 */
template<typename T>
void Cache<T>::L(bool ignore) {
    m_L.op(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    if (!ignore && m_debug) { m_callsToL += 1; }
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
    DTensor<T> yPrim(y, m_rowAxis, 0, m_sizePrim - 1);
    DTensor<T> yDual(y, m_rowAxis, m_sizePrim, m_sizeIterate - 1);
    yDual.deviceCopyTo(*m_d_workIterateDual);
    Ltr();
    m_d_workIteratePrim->deviceCopyTo(*m_d_workDotPrim);
    yPrim.deviceCopyTo(*m_d_workIteratePrim);
    L(true);
    m_d_workIterateDual->deviceCopyTo(*m_d_workDotDual);
    *m_d_workDot *= -m_data.stepSize();
    *m_d_workDot += y;
    return x.dotF(*m_d_workDot);
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
    m_d_iterateDual->deviceCopyTo(*m_d_workIterateDual);
    Ltr();
    *m_d_workIteratePrim *= -m_data.stepSize();
    *m_d_workIteratePrim += *m_d_iteratePrim;
}

/**
 * Compute proximal of f on primalBar
 */
template<typename T>
void Cache<T>::proximalPrimal() {
    proxRootS();
    projectPrimalWorkspaceOnDynamics();
    projectPrimalWorkspaceOnKernels();
    m_d_workIteratePrim->deviceCopyTo(*m_d_iterateCandidatePrim);  // Store primal
}

/**
 * Compute dualBar
 */
template<typename T>
void Cache<T>::modifyDual() {
    *m_d_workIteratePrim *= 2.;
    *m_d_workIteratePrim -= *m_d_iteratePrim;
    L();
    *m_d_workIterateDual *= m_data.stepSize();
    *m_d_workIterateDual += *m_d_iterateDual;
}

/**
 * Compute proximal of g* on dualBar
 */
template<typename T>
void Cache<T>::proximalDual() {
    *m_d_workIterateDual *= m_data.stepSizeRecip();
    translateSocs();
    m_d_workIterateDual->deviceCopyTo(*m_d_iterateCandidateDual);
    projectDualWorkspaceOnConstraints();
    *m_d_iterateCandidateDual -= *m_d_workIterateDual;
    *m_d_iterateCandidateDual *= m_data.stepSize();  // Store dual
}

/**
 * Compute one iteration of T(iterate) operator, nothing more.
 * Write output to `candidates`.
 */
template<typename T>
void Cache<T>::cpIter() {
    modifyPrimal();
    proximalPrimal();
    modifyDual();
    proximalDual();
}

template<typename T>
void Cache<T>::backup() {
    m_d_iterateCandidate->deviceCopyTo(*m_d_iterateBackup);
}

template<typename T>
void Cache<T>::restore() {
    m_d_iterateBackup->deviceCopyTo(*m_d_iterateCandidate);
}

/**
 * Compute residual.
 */
template<typename T>
void Cache<T>::computeResidual() {
    m_d_iterate->deviceCopyTo(*m_d_residual);
    *m_d_residual -= *m_d_iterateCandidate;
}

/**
 * Compute change of iterate.
 */
template<typename T>
void Cache<T>::computeDeltaIterate() {
    m_d_iterate->deviceCopyTo(*m_d_deltaIterate);
    *m_d_deltaIterate -= *m_d_iteratePrev;
}

/**
 * Compute change of residual.
 */
template<typename T>
void Cache<T>::computeDeltaResidual() {
    m_d_residual->deviceCopyTo(*m_d_deltaResidual);
    *m_d_deltaResidual -= *m_d_residualPrev;
}

/**
 * Save current iterates to prev.
 */
template<typename T>
void Cache<T>::saveToPrev() {
    m_d_iterate->deviceCopyTo(*m_d_iteratePrev);
    m_d_residual->deviceCopyTo(*m_d_residualPrev);
}

/**
 * Save candidates as accepted iterates.
 */
template<typename T>
void Cache<T>::acceptCandidate() {
    m_d_iterateCandidate->deviceCopyTo(*m_d_iterate);
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

    /**
     * 1. QR factorise `m_d_andResidualMatrix`.
     * 2. Compute least squares `m_d_andResidualMatrix \ m_d_residual`.
     * 4. Compute Anderson's direction.
     */
    if (idx >= m_andSize) {
        /* QR decomposition */
        m_d_andResidualMatrix->deviceCopyTo(*m_d_andQR);
        m_d_residual->deviceCopyTo(*m_d_andQRGammaFull);
        m_andQRFactor->factorise();
        if (m_debug) {
            DTensor<int> status = m_andQRFactor->info();
            if (status(0, 0, 0) != 0) {
                err << "[updateDirection] QR factorisation returned status code: " << m_status << "\n";
                throw std::invalid_argument(err.str());
            }
        }
        /* Least squares */
        m_andQRFactor->leastSquares(*m_d_andQRGammaFull);
        if (m_debug) {
            DTensor<int> status = m_andQRFactor->info();
            if (status(0, 0, 0) != 0) {
                err << "[updateDirection] QR least squares returned status code: " << m_status << "\n";
                throw std::invalid_argument(err.str());
            }
        }
        /* Compute new direction */
        m_d_direction->addAB(*m_d_andIterateMatrix, *m_d_andQRGamma, -1.);
        *m_d_direction -= *m_d_residual;
    } else {
        /* Use residual direction */
        m_d_residual->deviceCopyTo(*m_d_direction);
        *m_d_direction *= -1.;
    }
}

/**
 * Compute errors for termination check.
 */
template<typename T>
bool Cache<T>::computeError(size_t idx) {
    cudaDeviceSynchronize();  // DO NOT REMOVE !!!
    /* L(deltaIteratePrim) and L'(deltaIterateDual) */
    m_d_deltaIterateDual->deviceCopyTo(*m_d_workIterateDual);
    Ltr();
    m_d_workIteratePrim->deviceCopyTo(*m_d_ellDeltaIteratePrim);
    m_d_deltaIteratePrim->deviceCopyTo(*m_d_workIteratePrim);
    L(true);
    m_d_workIterateDual->deviceCopyTo(*m_d_ellDeltaIterateDual);
    /* -deltaIterate/step + ell(deltaIterate) */
    m_d_deltaIterate->deviceCopyTo(*m_d_workIterate);
    *m_d_workIterate *= -m_data.stepSizeRecip();
    *m_d_workIterate += *m_d_ellDeltaIterate;
    if (m_errInit) {
        m_tol = std::max(m_tolAbs, m_tolRel * m_d_workIterate->maxAbs());
        m_errInit = false;
        m_status = false;
    } else {
        m_errAbs = m_d_workIterate->maxAbs();
        m_status = (m_errAbs <= m_tol);
        if (m_debug) {
            m_cacheError1[idx] = m_d_workIteratePrim->maxAbs();
            m_cacheError2[idx] = m_d_workIterateDual->maxAbs();
            m_cacheCallsToL[idx] = m_callsToL;
            /* errPrim + L'(errDual) */
            m_d_workIteratePrim->deviceCopyTo(*m_d_err);
            Ltr();
            *m_d_err += *m_d_workIteratePrim;
            m_cacheError0[idx] = m_d_err->maxAbs();
        }
    }
    return m_status;
}

/**
 * CP algorithm.
 */
template<typename T>
int Cache<T>::runCp(std::vector<T> &initState, std::vector<T> *previousSolution) {
    /* Load initial state */
    initialiseState(initState);
    /* Load previous solution if given */
    initialisePrev(previousSolution);
    /* Reset error check */
    m_errInit = true;
    /* Run algorithm */
    for (size_t i = 0; i < m_maxOuterIters; i++) {
        /* Save iterate to prev */
        saveToPrev();
        /* Compute CP iteration */
        cpIter();
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
    /* Load initial state */
    initialiseState(initState);
    /* Load previous solution if given */
    initialisePrev(previousSolution);
    /* Reset error check */
    m_errInit = true;
    /* Initialise variables */
    size_t countK0 = 0;
    size_t countK1 = 0;
    size_t countK2 = 0;
    size_t countK2bt = 0;
    size_t countK3 = 0;
    T zeta = INFINITY;
    T w = INFINITY;
    T wSafe = INFINITY;
    T wTilde = INFINITY;
    T rho = 0;
    T tau = 0;
    /* Run */
    for (size_t iOut = 0; iOut < m_maxOuterIters; iOut++) {
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
        /* START */
        cpIter();
        backup();
        /* Compute residual */
        computeResidual();
        /* Compute residual norm */
        w = normM(*m_d_residual, *m_d_residual);
        /* Compute change of residual */
        computeDeltaResidual();
        /* Save iterate and residual to prev */
        saveToPrev();
        /* Compute direction */
        updateDirection(iOut);
        /* Blind update */
        if (w <= m_c0 * zeta && m_allowK0) {
            m_d_iteratePrev->deviceCopyTo(*m_d_iterateCandidate);
            *m_d_iterateCandidate += *m_d_direction;
            zeta = w;
            countK0 += 1;
            continue;  // K0
        }
        /* Line search on tau */
        tau = 1.;
        for (size_t iIn = 0; iIn < m_maxInnerIters; iIn++) {
            m_d_iteratePrev->deviceCopyTo(*m_d_iterate);
            m_d_direction->deviceCopyTo(*m_d_directionScaled);
            *m_d_directionScaled *= tau;
            *m_d_iterate += *m_d_directionScaled;
            cpIter();
            computeResidual();
            wTilde = normM(*m_d_residual, *m_d_residual);
            /* Educated update */
            if (w <= wSafe && wTilde <= m_c1 * w) {
                m_d_iterate->deviceCopyTo(*m_d_iterateCandidate);
                wSafe = wTilde + pow(m_c2, iOut);
                countK1 += 1;
                break;  // K1
            }
            /* Compute rho */
            *m_d_directionScaled *= -1.;
            *m_d_directionScaled += *m_d_residual;
            rho = dotM(*m_d_residual, *m_d_directionScaled);  // This is not the algo equation, but performs better.
            /* Safeguard update */
            if (rho >= m_sigma * wTilde * w) {
                *m_d_residual *= (m_lambda * rho / pow(wTilde, 2));
                m_d_iteratePrev->deviceCopyTo(*m_d_iterateCandidate);
                *m_d_iterateCandidate -= *m_d_residual;
                countK2 += 1;
                break;  // K2
            } else {
                tau *= m_beta;
                countK2bt += 1;
            }
            if (iIn >= m_maxInnerIters - 1) {
                restore();
                countK3 += 1;
            }
        }
    }
//    std::string n = "Sp";
//    printToJson(n);
    /* Return status */
    if (m_status) {
        if (m_debug) {
            std::cout << "\nConverged in " << m_countIterations << " outer iterations, to a tolerance of " << m_tol
                      << ", [K0: " << countK0
                      << ", K1: " << countK1
                      << ", K2: " << countK2
                      << ", bt: " << countK2bt
                      << ", K3: " << countK3
                      << "].\n";
        }
        return 0;
    } else {
        if (m_debug) {
            std::cout << "\nMax iterations (" << m_maxOuterIters << ") reached [K0: " << countK0
                      << ", K1: " << countK1
                      << ", K2: " << countK2
                      << ", bt: " << countK2bt
                      << ", K3: " << countK3
                      << "].\n";
        }
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
    doc.AddMember("sizePrim", m_sizePrim, doc.GetAllocator());
    doc.AddMember("sizeDual", m_sizeDual, doc.GetAllocator());
//    std::vector<T> solution(m_sizePrim);
//    m_d_iteratePrim->download(solution);
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
    std::ofstream of("/home/biggirl/Documents/remote_host/raocp-parallel/misc/cache" + file + ".json");
    of << json;
    if (!of.good()) throw std::runtime_error("[Cache::printToJson] Can't write the JSON string to the file!");
}

/**
 * Time vanilla CP algorithm with a parallelised cache
 */
template<typename T>
int Cache<T>::timeCp(std::vector<T> &initialState) {
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
T Cache<T>::timeSp(std::vector<T> &initialState) {
    const auto tick = std::chrono::high_resolution_clock::now();
    int status = runSpock(initialState);
    const auto tock = std::chrono::high_resolution_clock::now();
    if (status) {
        err << "Status error, not converged. [numStages=" << m_tree.numStages() << ", nx=nu=" << m_data.numStates()
            << "].\n";
        throw std::runtime_error(err.str());
    }
    T durationMilli = std::chrono::duration<T, std::milli>(tock - tick).count();
    return durationMilli;
}


#endif
