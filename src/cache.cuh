#ifndef __CACHE__
#define __CACHE__
#include "../include/stdgpu.h"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


__global__ void d_setPToId(real_t* matP, size_t node, size_t numRows);


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
class Cache {

private:
    ScenarioTree& m_tree;  ///< Previously created scenario tree of problem
    ProblemData& m_data;  ///< Previously created data of problem
    /** Solver data */
    real_t m_tol = 0;
    size_t m_maxIters = 0;
    size_t m_countIterations = 0;
    /** Primal data */
    size_t m_primSize = 0;
    DeviceVector<real_t> m_d_prim;
    DeviceVector<real_t> m_d_primMod;
    DeviceVector<real_t> m_d_primPrev;
    /** Dual data */
    size_t m_dualSize = 0;
    DeviceVector<real_t> m_d_dual;
    DeviceVector<real_t> m_d_dualMod;
    DeviceVector<real_t> m_d_dualPrev;
    /** Error data */
    DeviceVector<real_t> m_d_cacheError;
    /** Dynamics projection data */
    DeviceVector<real_t> m_d_P;
    DeviceVector<real_t> m_d_q;
    DeviceVector<real_t> m_d_K;
    DeviceVector<real_t> m_d_d;
    DeviceVector<real_t> m_d_choleskyLo;
    DeviceVector<real_t> m_d_choleskyUp;
    DeviceVector<real_t> m_d_dynamicsSum;  ///< A+BK
    /** Kernel projection data */
    DeviceVector<real_t> m_d_kernelConstraintMat;
    DeviceVector<real_t> m_d_nullSpaceMat;
    /** Methods */
    void offline_projection_setup();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree& tree, ProblemData& data, real_t tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        m_primSize = m_tree.numNodes() * m_data.numStates()  ///< States of all nodes
                + m_tree.numNonleafNodes() * m_data.numInputs()  ///< Inputs of all nonleaf nodes
                + m_tree.numNonleafNodes() * m_tree.numEvents()  ///< Y for all nonleaf nodes
                + m_tree.numNodes()  ///< T for all child nodes
                + m_tree.numNodes();  ///< S for all child nodes

        /** Allocate memory on device */
        m_d_prim.allocateOnDevice(m_primSize);
        m_d_primMod.allocateOnDevice(m_primSize);
        m_d_primPrev.allocateOnDevice(m_primSize);
        m_d_dual.allocateOnDevice(m_dualSize);
        m_d_dualMod.allocateOnDevice(m_dualSize);
        m_d_dualPrev.allocateOnDevice(m_dualSize);
        m_d_cacheError.allocateOnDevice(m_maxIters);

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
    if (initState.size() != m_data.numStates()) {
        std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                  << " but given " << initState.size() << " states" << std::endl;
        throw std::invalid_argument("Incorrect dimension of initial state");
    }
    m_d_prim.upload(initState);
}


void Cache::offline_projection_setup() {
    /** Offline: projection on dynamics */
    m_d_P.allocateOnDevice(m_data.numStates() * m_data.numStates() * m_tree.numNodes());
    m_d_q.allocateOnDevice(m_data.numStates() * m_tree.numNodes());
    m_d_K.allocateOnDevice(m_data.numInputs() * m_data.numStates() * m_tree.numNonleafNodes());
    m_d_d.allocateOnDevice(m_data.numInputs() * m_tree.numNonleafNodes());
    m_d_choleskyLo.allocateOnDevice(m_data.numInputs() * m_data.numInputs() * m_tree.numNonleafNodes());
    m_d_choleskyUp.allocateOnDevice(m_data.numInputs() * m_data.numInputs() * m_tree.numNonleafNodes());
    m_d_dynamicsSum.allocateOnDevice(m_data.numStates() * m_data.numStates() * m_tree.numNodes());
    for (size_t i=m_tree.numNonleafNodes(); i<m_tree.numNodes(); i++) {
        d_setPToId<<<m_data.numStates(), THREADS_PER_BLOCK>>>(m_d_P.get(), i, m_data.numStates());
    }
//    for (size_t i=m_tree.numNonleafNodes(); i<m_tree.numNodes(); i++) {
//        m_d_P = eye(state_size);
//    }
//    for (i in reversed(range(self.__num_nonleaf_nodes))) {
//        children_of_i = self.__raocp.tree.children_of(i);
//        sum_for_r = 0;
//        sum_for_k = 0;
//        for (j in children_of_i) {
//            sum_for_r = sum_for_r + \
//                                self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] @ \
//                                self.__raocp.control_dynamics_at_node(j);
//            sum_for_k = sum_for_k + \
//                                self.__raocp.control_dynamics_at_node(j).T @ self.__P[j] @ \
//                                self.__raocp.state_dynamics_at_node(j);
//        }
//        r_tilde = np.eye(self.__control_size) + sum_for_r;
//        self.__cholesky_data[i] = scipy.linalg.cho_factor(r_tilde);
//        self.__K[i] = scipy.linalg.cho_solve(self.__cholesky_data[i], -sum_for_k);
//        sum_for_p = 0;
//        for (j in children_of_i) {
//            self.__sum_of_dynamics[j] = self.__raocp.state_dynamics_at_node(j) \
//                + self.__raocp.control_dynamics_at_node(j) @ self.__K[i];
//            sum_for_p = sum_for_p + self.__sum_of_dynamics[j].T @ self.__P[j] @ self.__sum_of_dynamics[j];
//        }
//        self.__P[i] = np.eye(self.__state_size) + self.__K[i].T @ self.__K[i] + sum_for_p;
//    }
//
//    /** Offline: projection on kernel */
//    m_d_kernelConstraintMat.allocateOnDevice(_ * m_tree.numNonleafNodes());
//    m_d_nullSpaceMat.allocateOnDevice(_ * m_tree.numNonleafNodes());
//    for i in range(self.__num_nonleaf_nodes):
//        eye = np.eye(len(self.__raocp.tree.children_of(i)))
//        zeros = np.zeros((self.__raocp.risk_at_node(i).matrix_f.shape[1], eye.shape[0]))
//        row1 = np.hstack((self.__raocp.risk_at_node(i).matrix_e.T, -eye, -eye))
//        row2 = np.hstack((self.__raocp.risk_at_node(i).matrix_f.T, zeros, zeros))
//        self.__kernel_constraint_matrix[i] = np.vstack((row1, row2))
//        self.__null_space_matrix[i] = scipy.linalg.null_space(self.__kernel_constraint_matrix[i])
}


void Cache::vanillaCp() {
    for (size_t i=0; i<m_maxIters; i++) {
        /** update z_bar */
        /** update n_bar */
        /** update z */
        /** update n */
        /** compute error */
        /** check error */
        if (m_d_cacheError.fetchElementFromDevice(i) <= m_tol) {
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
    for (size_t i=0; i<m_primSize; i++) {
        std::cout << hostDataReal[i] << " ";
    }
    std::cout << std::endl;

    len = m_tree.numNodes() * m_data.numStates() * m_data.numStates();
    hostDataReal.resize(len);
    m_d_P.download(hostDataReal);
    std::cout << "P (from device): ";
    for (size_t i=0; i<len; i++) {
        std::cout << hostDataReal[i] << " ";
    }
    std::cout << std::endl;
}

#endif
