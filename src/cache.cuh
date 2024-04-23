#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


__host__ __device__ size_t getIdxMat(size_t node, size_t row, size_t col, size_t rows, size_t cols = 0);

__global__ void d_setMatToId(real_t *mat, size_t numRows, size_t node = 0);

__global__ void d_negate(real_t *data, size_t n, size_t node = 0);


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
class Cache {

private:
    ScenarioTree &m_tree;  ///< Previously created scenario tree of problem
    ProblemData &m_data;  ///< Previously created data of problem
    size_t m_numStates;
    size_t m_numInputs;
    /** Solver data */
    real_t m_tol = 0;
    size_t m_maxIters = 0;
    size_t m_countIterations = 0;
    /** Primal data */
    size_t m_primSize = 0;
    DTensor<real_t> m_d_prim;
    DTensor<real_t> m_d_primMod;
    DTensor<real_t> m_d_primPrev;
    /** Dual data */
    size_t m_dualSize = 0;
    DTensor<real_t> m_d_dual;
    DTensor<real_t> m_d_dualMod;
    DTensor<real_t> m_d_dualPrev;
    /** Error data */
    DTensor<real_t> m_d_cacheError;
    /** Dynamics projection data */
    DTensor<real_t> m_d_P;
    DTensor<real_t> m_d_q;
    DTensor<real_t> m_d_K;
    DTensor<real_t> m_d_d;
    DTensor<real_t> m_d_choleskyLo;
    DTensor<real_t> m_d_dynamicsSum;  ///< A+BK
    /** Kernel projection data */
    DTensor<real_t> m_d_kernelConstraintMat;
    DTensor<real_t> m_d_nullSpaceMat;

    /** Methods */
    void offline_projection_setup();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree &tree, ProblemData &data, real_t tol, size_t maxIters) :
            m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        m_numStates = m_data.numStates();
        m_numInputs = m_data.numInputs();
        m_primSize = m_tree.numNodes() * m_numStates  ///< States of all nodes
                     + m_tree.numNonleafNodes() * m_numInputs  ///< Inputs of all nonleaf nodes
                     + m_tree.numNonleafNodes() * m_tree.numEvents()  ///< Y for all nonleaf nodes
                     + m_tree.numNodes()  ///< T for all child nodes
                     + m_tree.numNodes();  ///< S for all child nodes

        /** Allocate memory on device */
        m_d_prim(m_primSize);
        m_d_primMod(m_primSize);
        m_d_primPrev(m_primSize);
        m_d_dual(m_dualSize);
        m_d_dualMod(m_dualSize);
        m_d_dualPrev(m_dualSize);
        m_d_cacheError(m_maxIters);

        /** Transfer array data to device */
//        m_d_tol.upload(vecTol);

        offline_projection_setup();
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
    void initialiseState(std::vector<real_t> initState);

    /**
     * Methods
     */

    /**
     * Algorithms
     */
    void vanillaCp();

    /**
     * Debugging
     */
    void print();
};


void Cache::initialiseState(std::vector<real_t> initState) {
    if (initState.size() != m_numStates) {
        std::cerr << "Error initialising state: problem setup for " << m_numStates
                  << " but given " << initState.size() << " states" << std::endl;
        throw std::invalid_argument("Incorrect dimension of initial state");
    }
    m_d_prim.upload(initState);
}


void Cache::offline_projection_setup() {
    /**
     * Offline: projection on dynamics
     */

    /** create identity matrices */
    DTensor<real_t> d_idInput(m_numInputs * m_numInputs);
    d_setMatToId<<<m_numInputs, THREADS_PER_BLOCK>>>(d_idInput.raw(), m_numInputs);

    DTensor<real_t> d_idState(m_numStates * m_numStates);
    d_setMatToId<<<m_numStates, THREADS_PER_BLOCK>>>(d_idState.raw(), m_numStates);

    /** setup Cholesky workspace */
    DTensor<real_t> d_workspace;
    DTensor<int> d_info(1);
//    gpuCholeskySetup(m_numInputs, d_workspace);

    /** allocate device memory for dynamics projection data*/
    m_d_P(m_numStates * m_numStates * m_tree.numNodes());
    m_d_q(m_numStates * m_tree.numNodes());
    m_d_K(m_numInputs * m_numStates * m_tree.numNonleafNodes());
    m_d_d(m_numInputs * m_tree.numNonleafNodes());
    m_d_choleskyLo(m_numInputs * m_numInputs * m_tree.numNonleafNodes());
    m_d_dynamicsSum(m_numStates * m_numStates * m_tree.numNodes());

    /** set all leaf P matrices to identity */
    for (size_t i = m_tree.numNonleafNodes(); i < m_tree.numNodes(); i++) {
        d_setMatToId<<<m_numStates, THREADS_PER_BLOCK>>>(m_d_P.raw(), m_numStates, i);
    }

//    for (size_t stage = m_tree.numStages() - 2; true; stage--) {  ///< we don't need `stage >= 0` because size_t
//        size_t nodeFrom = m_tree.nodeFrom().fetchElementFromDevice(stage);
//        size_t nodeTo = m_tree.nodeTo().fetchElementFromDevice(stage);
//
//        for (size_t node = nodeFrom; node <= nodeTo; node++) {
//            size_t chFrom = m_tree.childFrom().fetchElementFromDevice(node);
//            size_t chTo = m_tree.childTo().fetchElementFromDevice(node);
//
//            /** compute each K and Cholesky decomposition of R */
//            DTensor<real_t> d_forR(m_context, std::vector(m_numInputs * m_numInputs, 0.0));
//            DTensor<real_t> d_forK(m_context, std::vector(m_numInputs * m_numStates, 0.0));
//
//            for (size_t child = chFrom; child <= chTo; child++) {
//                DTensor<real_t> d_matA(m_data.stateDynamics(),
//                                            getIdxMat(child, 0, 0, m_numStates),
//                                            getIdxMat(child, m_numStates, m_numStates, m_numStates));
//                DTensor<real_t> d_matB(m_data.inputDynamics(),
//                                            getIdxMat(child, 0, 0, m_numStates, m_numInputs),
//                                            getIdxMat(child, m_numStates, m_numInputs, m_numStates, m_numInputs));
//                DTensor<real_t> d_matP(m_d_P,
//                                            getIdxMat(child, 0, 0, m_numStates),
//                                            getIdxMat(child, m_numStates, m_numStates, m_numStates));
//
//                DTensor<real_t> d_matBP(m_context, m_numInputs * m_numStates);
//                gpuMatMatMul(m_context, m_numInputs, m_numStates, m_numStates, d_matB, d_matP, d_matBP, true);
//
//                DTensor<real_t> d_matBPB(m_context, m_numInputs * m_numInputs);
//                gpuMatMatMul(m_context, m_numInputs, m_numStates, m_numInputs, d_matBP, d_matB, d_matBPB);
//
//                DTensor<real_t> d_matBPA(m_context, m_numInputs * m_numStates);
//                gpuMatMatMul(m_context, m_numInputs, m_numStates, m_numStates, d_matBP, d_matA, d_matBPA);
//                gpuMatAdd(m_context, m_numInputs, m_numInputs, d_matBPB, d_forR, d_forR);
//                gpuMatAdd(m_context, m_numInputs, m_numStates, d_matBPA, d_forK, d_forK);
//            }
//
//            gpuMatAdd(m_context, m_numInputs, m_numInputs, d_idInput, d_forR, d_forR);
//            gpuCholeskyFactor(m_context, m_numInputs, d_workspace, d_forR, true);
//
//            DTensor<real_t> d_matL(m_d_choleskyLo,
//                                        getIdxMat(node, 0, 0, m_numInputs),
//                                        getIdxMat(node, m_numInputs, m_numInputs, m_numInputs));
//            d_forR.deviceCopyTo(d_matL);
//            d_negate<<<m_numInputs * m_numStates, THREADS_PER_BLOCK>>>(d_forK.raw(), m_numInputs * m_numStates);
//
//            DTensor<real_t> d_matKt(m_context, m_numInputs * m_numStates);
//            gpuCholeskySolve(m_context, m_numInputs, m_numStates, d_matL, d_matKt, d_forK, true);
//
//            DTensor<real_t> d_matK(m_d_K,
//                                        getIdxMat(node, 0, 0, m_numInputs, m_numStates),
//                                        getIdxMat(node, m_numInputs, m_numStates, m_numInputs, m_numStates));
////            gpuMatT(m_context, m_numInputs, m_numStates, d_matKt, d_matK);
//
//            /** compute each P */
//            DTensor<real_t> d_forP(m_context, std::vector(m_numStates * m_numStates, 0.0));
//            for (size_t child = chFrom; child <= chTo; child++) {
//                DTensor<real_t> d_matA(m_data.stateDynamics(),
//                                            getIdxMat(child, 0, 0, m_numStates),
//                                            getIdxMat(child, m_numStates, m_numStates, m_numStates));
//                DTensor<real_t> d_matB(m_data.inputDynamics(),
//                                            getIdxMat(child, 0, 0, m_numStates, m_numInputs),
//                                            getIdxMat(child, m_numStates, m_numInputs, m_numStates, m_numInputs));
//                DTensor<real_t> d_matP(m_d_P,
//                                            getIdxMat(child, 0, 0, m_numStates),
//                                            getIdxMat(child, m_numStates, m_numStates, m_numStates));
//                DTensor<real_t> d_matBK(m_context, m_numStates * m_numStates);
//                gpuMatMatMul(m_context, m_numStates, m_numInputs, m_numStates, d_matB, d_matK, d_matBK);
//
//                DTensor<real_t> d_matD(m_d_dynamicsSum,
//                                            getIdxMat(child, 0, 0, m_numStates, m_numStates),
//                                            getIdxMat(child, m_numStates, m_numStates, m_numStates, m_numStates));
//                gpuMatAdd(m_context, m_numStates, m_numStates, d_matA, d_matBK, d_matD);
//
//                DTensor<real_t> d_matDPD(m_context, m_numStates * m_numStates);
//                gpuMatMatMul(m_context, m_numStates, m_numStates, m_numStates, d_matD, d_matP, d_matDPD, true);
//                gpuMatMatMul(m_context, m_numStates, m_numStates, m_numStates, d_matDPD, d_matD, d_matDPD);
//                gpuMatAdd(m_context, m_numStates, m_numStates, d_matDPD, d_forP, d_forP);
//            }
//
//            DTensor<real_t> d_matP(m_d_P,
//                                        getIdxMat(node, 0, 0, m_numStates),
//                                        getIdxMat(node, m_numStates, m_numStates, m_numStates));
//            DTensor<real_t> d_matKK(m_context, m_numStates * m_numStates);
//            gpuMatMatMul(m_context, m_numStates, m_numInputs, m_numStates, d_matK, d_matK, d_matKK, true);
//            gpuMatAdd(m_context, m_numStates, m_numStates, d_matKK, d_forP, d_forP);
//            gpuMatAdd(m_context, m_numStates, m_numStates, d_idState, d_forP, d_matP);
//        }
//        if (stage == 0) break;
//    }
//
//    /** Offline: projection on kernel */
//    m_d_kernelConstraintMat(_ * m_tree.numNonleafNodes());
//    m_d_nullSpaceMat(_ * m_tree.numNonleafNodes());
//    for i in range(self.__num_nonleaf_nodes):
//        eye = np.eye(len(self.__raocp.tree.children_of(i)))
//        zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
//        row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
//        row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
//        self.__kernel_constraint_matrix[i] = np.vstack((row1, row2))
//        self.__null_space_matrix[i] = scipy.linalg.null_space(self.__kernel_constraint_matrix[i])
}


void Cache::vanillaCp() {
    for (size_t i = 0; i < m_maxIters; i++) {
        /** update z_bar */
        /** update n_bar */
        /** update z */
        /** update n */
        /** compute error */
        /** check error */
        if (m_d_cacheError(i) <= m_tol) {
            m_countIterations = i;
            break;
        }
    }
}


void Cache::print() {
    std::cout << "Tolerance:      " << m_tol << std::endl;
    std::cout << "Max iterations: " << m_maxIters << std::endl;
    std::cout << "Num iterations: " << m_countIterations << std::endl;
    size_t len;
    std::vector<size_t> hostDataSize;
    std::vector<real_t> hostDataReal;

    hostDataReal.resize(m_primSize);
    m_d_prim.download(hostDataReal);
    std::cout << "Primal (from device): ";
    for (size_t i = 0; i < m_primSize; i++) {
        std::cout << hostDataReal[i] << " ";
    }
    std::cout << std::endl;

    len = m_tree.numNodes() * m_numStates * m_numStates;
    hostDataReal.resize(len);
    m_d_P.download(hostDataReal);
    std::cout << "P (from device): ";
    for (size_t i = 0; i < len; i++) {
        std::cout << hostDataReal[i] << " ";
    }
    std::cout << std::endl;
}

#endif
