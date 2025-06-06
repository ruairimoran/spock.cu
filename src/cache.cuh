#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"
#include "risks.cuh"
#include "operator.cuh"
#include <algorithm>
#include <chrono>
#include <filesystem>


TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyInTS(T *, T *, size_t, size_t, size_t, size_t *, size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyOutTS(T *, T *, size_t, size_t, size_t, size_t *, size_t *);

template<typename T>
class CacheTestData;

template<typename T>
void testInitialisingState(CacheTestData<T> &);

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
void testIsItReallyTheAdjoint(OperatorTestData<T> &, T);

template<typename T>
void testDotM(CacheTestData<T> &, T);


/**
 * Sanity check for non-finite values in vector.
 * Caution! For debugging only.
 */
template<typename T>
static void isFinite(DTensor<T> &d_vec) {
    std::vector<T> vec(d_vec.numEl());
    d_vec.download(vec);
    for (size_t i = 0; i < vec.size(); i++) {
        if (!std::isfinite(vec[i])) {
            std::cout << d_vec.tr();
            err << "[isFinite] DTensor has a non-finite entry at (" << i << ").\n";
            throw ERR;
        }
    }
}


/**
 * Cache status enum and string
 */
enum CacheStatus {
    notRun = -1,
    converged = 0,
    outOfIter = 1,
    outOfTime = 2
};

static const char *toString(int status) {
    switch (status) {
        case notRun:
            return "Cache has not been run!";
        case converged:
            return "Converged!";
        case outOfIter:
            return "Out of iterations!";
        case outOfTime:
            return "Out of time!";
        default:
            return "Unknown status.";
    }
}


/**
 * Cache of methods for proximal algorithms
 */
TEMPLATE_WITH_TYPE_T
class Cache {
protected:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
    LinearOperator<T> m_L = LinearOperator<T>(m_tree, m_data);  ///< Linear operator and its adjoint
    T m_tolAbs = 0.;
    T m_tolRel = 0.;
    T m_tol = 0.;
    T m_errAbs = 0.;
    T m_maxTimeSecs = 0.;
    size_t m_maxOuterIters = 0;
    size_t m_andBuff = 0;
    size_t m_countIterations = 0;
    size_t m_rowAxis = 0;
    size_t m_colAxis = 1;
    size_t m_matAxis = 2;
    size_t m_callsToL = 0;
    size_t m_computeErrorIters = 25;
    bool m_debug = false;
    bool m_errInit = false;  ///< Whether to initialise tolerances
    bool m_status = false;  ///< General status use
    int m_exitCode = notRun;  ///< Algorithm exit code
    std::chrono::high_resolution_clock::time_point m_timeStart;
    T m_timeElapsed = 0.;
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
    std::unique_ptr<DTensor<T>> m_d_iterateBackup = nullptr;
    std::unique_ptr<DTensor<T>> m_d_err = nullptr;
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
    std::unique_ptr<DTensor<T>> m_d_ellResidual = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellResidualPrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ellResidualDual = nullptr;
    /* Workspaces */
    std::vector<T> m_sizeIterateOnes;
    std::vector<T> m_initState;
    std::unique_ptr<DTensor<T>> m_d_initState = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workIterate = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workIteratePrim = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workIterateDual = nullptr;
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
    std::unique_ptr<DTensor<T>> m_d_v = nullptr;
    std::unique_ptr<DTensor<T>> m_d_vi = nullptr;
    std::unique_ptr<DTensor<T>> m_d_input = nullptr;
    /* Projections */
    std::unique_ptr<NonnegativeOrthantCone<T>> m_nnoc = nullptr;
    /* Caches */
    std::vector<size_t> m_cacheCallsToL;
    std::vector<T> m_cacheError0;
    std::vector<T> m_cacheError1;
    std::vector<T> m_cacheError2;
    std::vector<T> m_cacheError3;
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

    void initialiseIter(std::vector<T> *);

    void initialisePrev(std::vector<T> *);

    void initialiseState(std::vector<T> &initState);

    void proxRootS();

    void projectPrimalWorkspaceOnDynamics();

    void projectPrimalWorkspaceOnKernels();

    void translateSocs();

    void projectDualWorkspaceOnConstraints();

    void L(bool = false);

    void Ltr(bool = false);

    T dotM(DTensor<T> &, DTensor<T> &);

    T normM(DTensor<T> &, DTensor<T> &);

    void modifyPrimal();

    void proximalPrimal();

    void modifyDual();

    void proximalDual();

    void iter();

    void backup();

    void restore();

    void saveToPrev();

    void acceptCandidate();

    void computeResidual();

    void computeDeltaIterate();

    void computeDeltaResidual();

    void updateDirection(size_t);

    void computeError(size_t);

    void printToJson(std::string &);

public:
    /**
     * Constructor
     * Caution! Preferably use builder pattern.
     */
    Cache(ScenarioTree<T> &tree,
          ProblemData<T> &data,
          T absTol = 1e-3,
          T relTol = 0.,
          T maxTimeSecs = 0.,
          size_t maxOuterIters = 1000,
          size_t maxInnerIters = 8,
          size_t andBuff = 3,
          bool debug = false) :
        m_tree(tree), m_data(data), m_tolAbs(absTol), m_tolRel(relTol), m_maxTimeSecs(maxTimeSecs),
        m_maxOuterIters(maxOuterIters), m_maxInnerIters(maxInnerIters), m_andBuff(andBuff), m_debug(debug) {
        /* Tolerances */
        if (m_data.preconditioned()) {
            T tolScale = *std::min_element(m_data.scaling().begin(), m_data.scaling().end());
            m_tolAbs *= tolScale;
            m_tolRel *= tolScale;
        }
        /* Sizes */
        initialiseSizes();
        /* Allocate memory on host */
        m_initState = std::vector<T>(m_tree.numStates());
        m_sizeIterateOnes = std::vector<T>(m_sizeIterate, 1.);
        if (m_debug) {
            m_cacheCallsToL = std::vector<size_t>(m_maxOuterIters);
            m_cacheError0 = std::vector<T>(m_maxOuterIters);
            m_cacheError1 = std::vector<T>(m_maxOuterIters);
            m_cacheError2 = std::vector<T>(m_maxOuterIters);
            m_cacheError3 = std::vector<T>(m_maxOuterIters);
            m_cacheDeltaPrim = std::vector<T>(m_maxOuterIters);
            m_cacheDeltaDual = std::vector<T>(m_maxOuterIters);
            m_cacheNrmLtrDeltaDual = std::vector<T>(m_maxOuterIters);
            m_cacheDistDeltaDual = std::vector<T>(m_maxOuterIters);
            m_cacheSuppDeltaDual = std::vector<T>(m_maxOuterIters);
        }
        /* Allocate memory on device */
        m_d_initState = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, 1, true);
        m_d_iterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_iteratePrev = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_iterateCandidate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_iterateBackup = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_deltaIterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_ellResidual = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_deltaResidual = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_residual = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_residualPrev = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_err = std::make_unique<DTensor<T>>(m_sizePrim, 1, 1, true);
        m_d_andIterateMatrix = std::make_unique<DTensor<T>>(m_sizeIterate, m_andBuff, 1, true);
        m_d_andResidualMatrix = std::make_unique<DTensor<T>>(m_sizeIterate, m_andBuff, 1, true);
        m_d_andQR = std::make_unique<DTensor<T>>(m_sizeIterate, m_andBuff, 1, true);
        m_d_andQRGammaFull = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_direction = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_directionScaled = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_workIterate = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        m_d_workYTS = std::make_unique<DTensor<T>>(m_data.nullDim(), 1, m_tree.numNonleafNodes(), true);
        m_d_workDot = std::make_unique<DTensor<T>>(m_sizeIterate, 1, 1, true);
        /* Slice and reshape tensors */
        reshape();
        /* Initialise projectable objects */
        initialiseProjectable();
    }

    ~Cache() = default;

    int runCp(std::vector<T> &, std::vector<T> * = nullptr);

    int runSpock(std::vector<T> &, std::vector<T> * = nullptr);

    int status() { return m_exitCode; }

    size_t solveIter() { return m_countIterations; }

    T solveTime() { return m_timeElapsed; }

    /**
     * Debug functions (slow).
     */

    size_t sizePrimal() { return m_sizePrim; }

    size_t sizeDual() { return m_sizeDual; }

    DTensor<T> getDual(DTensor<T> &primal) {
        primal.deviceCopyTo(*m_d_workIteratePrim);
        modifyDual();
        proximalDual();
        return *m_d_iterateCandidateDual;
    }

    DTensor<T> &solution() { return *m_d_iterate; }

    DTensor<T> &solutionPrim() { return *m_d_iteratePrim; }

    DTensor<T> &solutionDual() { return *m_d_iterateDual; }

    void reset();

    std::vector<T> inputs() {
        m_d_iteratePrim->deviceCopyTo(*m_d_workIteratePrim);
        std::vector<T> inputs(m_sizeU);
        m_d_u->download(inputs);
        if (m_data.preconditioned()) {
            for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
                for (size_t ele = 0; ele < m_tree.numInputs(); ele++) {
                    inputs[node * m_tree.numInputs() + ele] /= m_data.scaling()[ele + m_tree.numStates()];
                }
            }
        }
        return inputs;
    }

    std::vector<T> states() {
        m_d_iteratePrim->deviceCopyTo(*m_d_workIteratePrim);
        std::vector<T> states(m_sizeX);
        m_d_x->download(states);
        if (m_data.preconditioned()) {
            for (size_t node = 0; node < m_tree.numNodes(); node++) {
                for (size_t ele = 0; ele < m_tree.numStates(); ele++) {
                    states[node * m_tree.numStates() + ele] /= m_data.scaling()[ele];
                }
            }
        }
        return states;
    }

    /**
     * Test functions. As a friend, they can access protected members.
     */
    friend void testInitialisingState<>(CacheTestData<T> &);

    friend void testKernelProjectionOnline<>(CacheTestData<T> &, T);

    friend void testKernelProjectionOnlineOrthogonality<>(CacheTestData<T> &, T);

    friend void testOperator<>(OperatorTestData<T> &, T);

    friend void testAdjoint<>(OperatorTestData<T> &, T);

    friend void testIsItReallyTheAdjoint<>(OperatorTestData<T> &, T);

    friend void testDotM<>(CacheTestData<T> &, T);
};

/**
 * Reset iterates and workspaces to zero.
 * Caution! For offline use only.
 */
template<typename T>
void Cache<T>::reset() {
    m_exitCode = notRun;
    m_data.dynamics()->resetWorkspace();
    m_L.resetWorkspace();
    /* Create zero vectors */
    std::vector<T> numStates(m_tree.numStates(), 0.);
    std::vector<T> sizeIterate(m_sizeIterate, 0.);
    std::vector<T> sizePrim(m_sizePrim, 0.);
    std::vector<T> sizeDual(m_sizeDual, 0.);
    std::vector<T> sizeIterateSizeAnd(m_sizeIterate * m_andBuff, 0.);
    std::vector<T> nullDimNumNonleafNodes(m_data.nullDim() * m_tree.numNonleafNodes(), 0.);
    /* Zero all cached data */
    std::fill(m_initState.begin(), m_initState.end(), 0.);
    std::fill(m_cacheCallsToL.begin(), m_cacheCallsToL.end(), 0.);
    std::fill(m_cacheError0.begin(), m_cacheError0.end(), 0.);
    std::fill(m_cacheError1.begin(), m_cacheError1.end(), 0.);
    std::fill(m_cacheError2.begin(), m_cacheError2.end(), 0.);
    std::fill(m_cacheDeltaPrim.begin(), m_cacheDeltaPrim.end(), 0.);
    std::fill(m_cacheDeltaDual.begin(), m_cacheDeltaDual.end(), 0.);
    std::fill(m_cacheNrmLtrDeltaDual.begin(), m_cacheNrmLtrDeltaDual.end(), 0.);
    std::fill(m_cacheDistDeltaDual.begin(), m_cacheDistDeltaDual.end(), 0.);
    std::fill(m_cacheSuppDeltaDual.begin(), m_cacheSuppDeltaDual.end(), 0.);
    m_d_initState->upload(numStates);
    m_d_iterate->upload(sizeIterate);
    m_d_iteratePrev->upload(sizeIterate);
    m_d_iterateCandidate->upload(sizeIterate);
    m_d_iterateBackup->upload(sizeIterate);
    m_d_deltaIterate->upload(sizeIterate);
    m_d_ellResidual->upload(sizeIterate);
    m_d_deltaResidual->upload(sizeIterate);
    m_d_residual->upload(sizeIterate);
    m_d_residualPrev->upload(sizeIterate);
    m_d_err->upload(sizePrim);
    m_d_andIterateMatrix->upload(sizeIterateSizeAnd);
    m_d_andResidualMatrix->upload(sizeIterateSizeAnd);
    m_d_andQR->upload(sizeIterateSizeAnd);
    m_d_andQRGammaFull->upload(sizeIterate);
    m_d_direction->upload(sizeIterate);
    m_d_directionScaled->upload(sizeIterate);
    m_d_workIterate->upload(sizeIterate);
    m_d_workYTS->upload(nullDimNumNonleafNodes);
    m_d_workDot->upload(sizeIterate);
}

template<typename T>
void Cache<T>::initialiseSizes() {
    m_sizeU = m_tree.numNonleafNodes() * m_tree.numInputs();  ///< Inputs of all nonleaf nodes
    m_sizeX = m_tree.numNodes() * m_tree.numStates();  ///< States of all nodes
    m_sizeY = m_tree.numNonleafNodes() * m_data.yDim();  ///< Y for all nonleaf nodes
    m_sizeT = m_tree.numNodes();  ///< T for all child nodes
    m_sizeS = m_tree.numNodes();  ///< S for all child nodes
    m_sizePrim = m_sizeU + m_sizeX + m_sizeY + m_sizeT + m_sizeS;
    m_sizeI = m_tree.numNonleafNodes() * m_data.yDim();
    m_sizeII = m_tree.numNonleafNodes();
    m_sizeIII = m_data.nonleafConstraint()->dimension();
    m_sizeIV = m_data.nonleafCost()->dim();
    m_sizeV = m_data.leafConstraint()->dimension();
    m_sizeVI = m_data.leafCost()->dim();
    m_sizeDual = m_sizeI + m_sizeII + m_sizeIII + m_sizeIV + m_sizeV + m_sizeVI;
    m_sizeIterate = m_sizePrim + m_sizeDual;
}

template<typename T>
void Cache<T>::reshape() {
    m_d_input = std::make_unique<DTensor<T>>(*m_d_iterate, m_rowAxis, 0, m_tree.numInputs() - 1);
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
    m_d_ellResidualPrim = std::make_unique<DTensor<T>>(*m_d_ellResidual, m_rowAxis, 0, m_sizePrim - 1);
    m_d_ellResidualDual = std::make_unique<DTensor<T>>(*m_d_ellResidual, m_rowAxis, m_sizePrim,
                                                       m_sizeIterate - 1);
    m_d_andIterateMatrixLeft = std::make_unique<DTensor<T>>(*m_d_andIterateMatrix, m_colAxis, 0, m_andBuff - 2);
    m_d_andIterateMatrixRight = std::make_unique<DTensor<T>>(*m_d_andIterateMatrix, m_colAxis, 1, m_andBuff - 1);
    m_d_andIterateMatrixCol0 = std::make_unique<DTensor<T>>(*m_d_andIterateMatrix, m_colAxis, 0, 0);
    m_d_andResidualMatrixLeft = std::make_unique<DTensor<T>>(*m_d_andResidualMatrix, m_colAxis, 0, m_andBuff - 2);
    m_d_andResidualMatrixRight = std::make_unique<DTensor<T>>(*m_d_andResidualMatrix, m_colAxis, 1, m_andBuff - 1);
    m_d_andResidualMatrixCol0 = std::make_unique<DTensor<T>>(*m_d_andResidualMatrix, m_colAxis, 0, 0);
    m_d_andQRGamma = std::make_unique<DTensor<T>>(*m_d_andQRGammaFull, m_rowAxis, 0, m_andBuff - 1);
    reshapePrimalWorkspace();
    reshapeDualWorkspace();
}

template<typename T>
void Cache<T>::reshapePrimalWorkspace() {
    size_t rowAxis = 0;
    m_d_workIteratePrim = std::make_unique<DTensor<T>>(*m_d_workIterate, rowAxis, 0, m_sizePrim - 1);
    size_t start = 0;
    m_d_u = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeU - 1);
    m_d_u->reshape(m_tree.numInputs(), 1, m_tree.numNonleafNodes());
    start += m_sizeU;
    m_d_x = std::make_unique<DTensor<T>>(*m_d_workIteratePrim, rowAxis, start, start + m_sizeX - 1);
    m_d_x->reshape(m_tree.numStates(), 1, m_tree.numNodes());
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
        m_data.nonleafConstraint()->reshape(*m_d_iii);
    }
    start += m_sizeIII;
    m_d_iv = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeIV - 1);
    m_d_iv->reshape(m_data.nonleafCost()->dimPerNode(), 1, m_data.nonleafCost()->numNodes());
    m_data.nonleafCost()->reshape(*m_d_workIterateDual, start, start + m_sizeIV - 1);
    start += m_sizeIV;
    if (m_sizeV) {
        m_d_v = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeV - 1);
        m_data.leafConstraint()->reshape(*m_d_v);
    }
    start += m_sizeV;
    m_d_vi = std::make_unique<DTensor<T>>(*m_d_workIterateDual, m_rowAxis, start, start + m_sizeVI - 1);
    m_d_vi->reshape(m_data.leafCost()->dimPerNode(), 1, m_data.leafCost()->numNodes());
    m_data.leafCost()->reshape(*m_d_workIterateDual, start, start + m_sizeVI - 1);
}

template<typename T>
void Cache<T>::initialiseProjectable() {
    /* Dual II */
    m_nnoc = std::make_unique<NonnegativeOrthantCone<T>>(m_tree.numNonleafNodes());
    /* QR */
    m_andQRFactor = std::make_unique<QRFactoriser<T>>(*m_d_andQR);
}

template<typename T>
void Cache<T>::initialiseState(std::vector<T> &initState) {
    /* Set initial state */
    if (initState.size() != m_tree.numStates()) {
        err << "[initialiseState] Error initialising state: problem setup for " << m_tree.numStates()
            << " but given " << initState.size() << " states" << "\n";
        throw ERR;
    }
    if (m_data.preconditioned()) {
        for (size_t i = 0; i < m_tree.numStates(); i++) { m_initState[i] = initState[i] * m_data.scaling()[i]; }
        m_d_initState->upload(m_initState);
    } else {
        m_d_initState->upload(initState);
    }
}

/**
 * Initialise iterate.
 * - If given, load previous solution.
 */
template<typename T>
void Cache<T>::initialiseIter(std::vector<T> *previousSolution) {
    if (previousSolution) {
        m_d_iterate->upload(*previousSolution);
    }
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
        m_d_iteratePrev->upload(m_sizeIterateOnes);
        m_d_residualPrev->upload(m_sizeIterateOnes);
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
    m_data.dynamics()->project(*m_d_initState, *m_d_x, *m_d_u);
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
    m_data.nonleafCost()->translate(*m_d_iv);
    m_data.leafCost()->translate(*m_d_vi);
}

template<typename T>
void Cache<T>::projectDualWorkspaceOnConstraints() {
    /* I */
    m_data.risk()->projectDual(*m_d_iNnoc);
    /* II */
    m_nnoc->project(*m_d_ii);
    /* III */
    m_data.nonleafConstraint()->project(*m_d_iii);
    /* IV */
    m_data.nonleafCost()->project();
    /* V */
    m_data.leafConstraint()->project(*m_d_v);
    /* VI */
    m_data.leafCost()->project();
}

/**
 * Call operator L on workspace
 */
template<typename T>
void Cache<T>::L(bool add) {
    m_L.op(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    if (add && m_debug) { m_callsToL += 1; }
}

/**
 * Call operator L' on workspace
 */
template<typename T>
void Cache<T>::Ltr(bool add) {
    m_L.adj(*m_d_u, *m_d_x, *m_d_y, *m_d_t, *m_d_s, *m_d_i, *m_d_ii, *m_d_iii, *m_d_iv, *m_d_v, *m_d_vi);
    if (add && m_debug) { m_callsToL += 1; }
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
    L();
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
    Ltr(true);
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
    L(true);
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
void Cache<T>::iter() {
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
    if (idx >= m_andBuff) {
        /* QR decomposition */
        m_d_andResidualMatrix->deviceCopyTo(*m_d_andQR);
        m_d_residual->deviceCopyTo(*m_d_andQRGammaFull);
        m_andQRFactor->factorise();
        if (m_debug) {
            DTensor<int> status = m_andQRFactor->info();
            if (status(0, 0, 0) != 0) {
                err << "[updateDirection] QR factorisation returned status code: " << m_status << "\n";
                throw ERR;
            }
        }
        /* Least squares */
        m_andQRFactor->leastSquares(*m_d_andQRGammaFull);
        if (m_debug) {
            DTensor<int> status = m_andQRFactor->info();
            if (status(0, 0, 0) != 0) {
                err << "[updateDirection] QR least squares returned status code: " << m_status << "\n";
                throw ERR;
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
void Cache<T>::computeError(size_t idx) {
    cudaDeviceSynchronize();  // DO NOT REMOVE !!!
    if (m_debug) isFinite(*m_d_iterateCandidate);
    computeResidual();
    if (idx % m_computeErrorIters == 0) {
        /* L(residualPrim) and L'(residualDual) */
        m_d_residualDual->deviceCopyTo(*m_d_workIterateDual);
        Ltr();
        m_d_workIteratePrim->deviceCopyTo(*m_d_ellResidualPrim);
        m_d_residualPrim->deviceCopyTo(*m_d_workIteratePrim);
        L();
        m_d_workIterateDual->deviceCopyTo(*m_d_ellResidualDual);
        /* residual/step - ell(residual) */
        m_d_residual->deviceCopyTo(*m_d_workIterate);
        *m_d_workIterate *= m_data.stepSizeRecip();
        *m_d_workIterate -= *m_d_ellResidual;
        if (m_errInit) {
            m_tol = std::max(m_tolAbs, m_tolRel * m_d_workIterate->maxAbs());
            m_exitCode = notRun;
            m_errInit = false;
        } else {
            m_errAbs = m_d_workIterate->maxAbs();
            if (m_errAbs <= m_tol) m_exitCode = converged;
            if (m_maxOuterIters) {
                if (idx >= m_maxOuterIters) m_exitCode = outOfIter;
            }
            if (m_maxTimeSecs && idx % 1000 == 0) {
                m_timeElapsed = std::chrono::duration<T>(
                    std::chrono::high_resolution_clock::now() - m_timeStart).count();
                if (m_timeElapsed >= m_maxTimeSecs) m_exitCode = outOfTime;
            }
            if (m_exitCode != notRun) {
                m_countIterations = idx;
                m_timeElapsed = std::chrono::duration<T>(
                    std::chrono::high_resolution_clock::now() - m_timeStart).count();
            }
        }
        if (m_debug) {
            m_cacheError1[idx] = m_d_workIteratePrim->maxAbs();
            m_cacheError2[idx] = m_d_workIterateDual->maxAbs();
            m_cacheCallsToL[idx] = m_callsToL;
            /* new_iter - T(old_iter) */
            m_cacheError3[idx] = (*m_d_iterate - *m_d_iterateCandidate).normF();
            /* cand_prim - L'(cand_dual) */
            m_d_iterateCandidateDual->deviceCopyTo(*m_d_workIterateDual);
            Ltr();
            *m_d_err *= -1.;
            *m_d_err += *m_d_iterateCandidatePrim;
            m_cacheError0[idx] = m_d_err->normF();
        }
    }
}

/**
 * CP algorithm.
 */
template<typename T>
int Cache<T>::runCp(std::vector<T> &initState, std::vector<T> *previousSolution) {
    m_timeStart = std::chrono::high_resolution_clock::now();
    /* Load initial state */
    initialiseState(initState);
    /* Load previous solution if given */
    initialiseIter(previousSolution);
    /* Reset error check */
    m_errInit = true;
    /* Run algorithm */
    for (size_t i = 0; i < SIZE_MAX; i++) {
        /* Compute CP iteration */
        iter();
        /* Check error */
        computeError(i);
        /* Save candidate to accepted iterate */
        acceptCandidate();
        /* Break if termination criteria met */
        if (m_exitCode != notRun) {
            break;
        }
    }
    if (m_debug) {
        std::string n = "Cp";
        printToJson(n);
        if (m_exitCode == converged) {
            std::cout << "\nConverged in " << m_countIterations << " iterations, to a tolerance of " << m_tol << "\n";
        } else if (m_exitCode == outOfIter) {
            std::cout << "\nOut of iterations (" << m_maxOuterIters << " iters).\n";
        } else if (m_exitCode == outOfTime) {
            std::cout << "\nOut of time (" << m_maxTimeSecs << " secs).\n";
        } else {
            std::cout << "\nExited.\n";
        }
    }
    return m_exitCode;
}

/**
 * SPOCK algorithm.
 */
template<typename T>
int Cache<T>::runSpock(std::vector<T> &initState, std::vector<T> *previousSolution) {
    m_timeStart = std::chrono::high_resolution_clock::now();
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
    for (size_t iOut = 0; iOut < SIZE_MAX; iOut++) {
        if (iOut != 0) {
            /* Accept new iterate */
            acceptCandidate();
        }
        /* START */
        iter();
        /* Check error (residual computed internally) */
        computeError(iOut);
        if (m_exitCode != notRun) {
            acceptCandidate();
            break;
        }
        /* Backup candidate */
        backup();
        /* Compute residual norm */
        w = normM(*m_d_residual, *m_d_residual);
        /* Compute change of iterate */
        computeDeltaIterate();
        /* Compute change of residual */
        computeDeltaResidual();
        /* Save iterate and residual to prev */
        saveToPrev();
        /* Compute direction */
        updateDirection(iOut);
        /* Blind update */
        if (w <= m_c0 * zeta) {
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
            iter();
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
            rho = pow(wTilde, 2) - dotM(*m_d_residual, *m_d_directionScaled);  // This is the algo equation.
//            *m_d_directionScaled *= -1.;  // This is...
//            *m_d_directionScaled += *m_d_residual;  // not the algo equation, but...
//            rho = dotM(*m_d_residual, *m_d_directionScaled);  //  performs well.
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
    if (m_debug) {
        std::string n = "Sp";
        printToJson(n);
        if (m_exitCode == converged) {
            std::cout << "\nConverged in " << m_countIterations << " iters and "
                      << m_timeElapsed << " secs, to a tolerance of " << m_tol
                      << ", [K0: " << countK0
                      << ", K1: " << countK1
                      << ", K2: " << countK2
                      << ", bt: " << countK2bt
                      << ", K3: " << countK3
                      << "].\n";
        } else if (m_exitCode == outOfIter) {
            std::cout << "\nMax iterations (" << m_maxOuterIters << ") reached [K0: " << countK0
                      << ", K1: " << countK1
                      << ", K2: " << countK2
                      << ", bt: " << countK2bt
                      << ", K3: " << countK3
                      << "].\n";
        } else if (m_exitCode == outOfTime) {
            std::cout << "\nMax time (" << m_maxTimeSecs << "s) reached [K0: " << countK0
                      << ", K1: " << countK1
                      << ", K2: " << countK2
                      << ", bt: " << countK2bt
                      << ", K3: " << countK3
                      << "].\n";
        } else {
            std::cout << "\nExited.\n";
        }
    }
    return m_exitCode;
}

/**
 * Builder pattern for Cache
 */
TEMPLATE_WITH_TYPE_T
class CacheBuilder {
private:
    ScenarioTree<T> &m_tree;
    ProblemData<T> &m_data;
    T m_tolAbs = 0.;
    T m_tolRel = 0.;
    T m_maxTimeSecs = 0.;
    size_t m_maxOuterIters = 0;
    size_t m_maxInnerIters = 0;
    size_t m_andBuff = 0;
    bool m_debug = false;

public:
    /**
     * Constructor with default values
     */
    CacheBuilder(ScenarioTree<T> &tree, ProblemData<T> &data) :
        m_tree(tree),
        m_data(data),
        m_tolAbs(1e-3),
        m_tolRel(0.),
        m_maxTimeSecs(0.),
        m_maxOuterIters(0),
        m_maxInnerIters(8),
        m_andBuff(3),
        m_debug(false) {};

    /**
     * Setters
     */
    CacheBuilder<T> &toleranceAbsolute(T tol) {
        m_tolAbs = tol;
        return *this;
    }

    CacheBuilder<T> &toleranceRelative(T tol) {
        m_tolRel = tol;
        return *this;
    }

    CacheBuilder<T> &tol(T tol) {
        m_tolAbs = tol;
        m_tolRel = tol;
        return *this;
    }

    CacheBuilder<T> &maxTimeSecs(T time) {
        m_maxTimeSecs = time;
        return *this;
    }

    CacheBuilder<T> &maxIters(size_t iters) {
        m_maxOuterIters = iters;
        return *this;
    }

    CacheBuilder<T> &maxItersInner(size_t iters) {
        m_maxInnerIters = iters;
        return *this;
    }

    CacheBuilder<T> &andersonBuffer(size_t buffer) {
        m_andBuff = buffer;
        return *this;
    }

    CacheBuilder<T> &enableDebug(bool enable = true) {
        if (enable && m_maxOuterIters == 0) {
            m_maxOuterIters = 1000;
        }
        m_debug = enable;
        return *this;
    }

    /**
     * Build Cache
     */
    Cache<T> build() {
        return Cache<T>(
            m_tree,
            m_data,
            m_tolAbs,
            m_tolRel,
            m_maxTimeSecs,
            m_maxOuterIters,
            m_maxInnerIters,
            m_andBuff,
            m_debug
        );
    }

    /**
     * Build unique_ptr to Cache
     */
    std::unique_ptr<Cache<T>> make_unique() {
        return std::make_unique<Cache<T>>(
            m_tree,
            m_data,
            m_tolAbs,
            m_tolRel,
            m_maxTimeSecs,
            m_maxOuterIters,
            m_maxInnerIters,
            m_andBuff,
            m_debug
        );
    }
};

/**
 * Add a vector to a .json file
 */
template<typename T>
static void
addArray(rapidjson::Document &doc, std::string const &name, std::vector<T> &vec) {
    for (const T &value: vec) {
        if (!std::isfinite(value)) {
            err << "[Cache::addArray] array (" << name << ") has entries that are not finite!\n";
            throw ERR;
        }
    }
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
    doc.AddMember("horizon", m_tree.numStages() - 1, doc.GetAllocator());
    auto idx = find(m_tree.childMax().begin(), m_tree.childMax().end(), 1);
    doc.AddMember("branchFactor", idx - m_tree.childMax().begin(), doc.GetAllocator());
    doc.AddMember("numEvents", m_tree.numEvents(), doc.GetAllocator());
    doc.AddMember("numStates", m_tree.numStates(), doc.GetAllocator());
    doc.AddMember("numInputs", m_tree.numInputs(), doc.GetAllocator());
    doc.AddMember("sizeCache", m_maxOuterIters, doc.GetAllocator());
    addArray(doc, "callsL", m_cacheCallsToL);
    addArray(doc, "err0", m_cacheError0);
    addArray(doc, "err1", m_cacheError1);
    addArray(doc, "err2", m_cacheError2);
    addArray(doc, "err3", m_cacheError3);
    std::vector<T> x = this->states();
    addArray(doc, "states", x);
    std::vector<T> u = this->inputs();
    addArray(doc, "inputs", u);
    typedef rapidjson::GenericStringBuffer<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>> StringBuffer;
    StringBuffer buffer(&doc.GetAllocator());
    rapidjson::Writer<StringBuffer> writer(buffer, reinterpret_cast<rapidjson::CrtAllocator *>(&doc.GetAllocator()));
    doc.Accept(writer);
    std::string json(buffer.GetString(), buffer.GetSize());
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string filename = "cache" + file + ".json";
    std::filesystem::path logFilePath = cwd / "log" / filename;
    std::filesystem::create_directories(logFilePath.parent_path());
    std::ofstream logFile(logFilePath);
    if (!logFile.is_open()) throw std::runtime_error("[Cache::printToJson] Failed to open file for writing.\n");
    logFile << json;
    logFile.close();
    if (!logFile.good()) throw std::runtime_error("[Cache::printToJson] Can't write the JSON string to the file.\n");
}


#endif
