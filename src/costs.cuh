#ifndef COSTS_CUH
#define COSTS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"


/**
 * Base class for costs
 */
template<typename T>
class Cost {

protected:
    size_t m_matAxis = 2;
    ScenarioTree<T> &m_tree;
    /* Dynamics data */
    std::unique_ptr<DTensor<T>> m_d_BTr = nullptr;  ///< input dynamics, B'
    std::unique_ptr<DTensor<T>> m_d_AB = nullptr;  ///< combined dynamics, [A B]
    std::unique_ptr<DTensor<T>> m_d_e = nullptr;  ///< constant dynamics, e
    bool m_affine = false;
    /* Projection data */
    std::unique_ptr<DTensor<T>> m_d_K = nullptr;
    std::unique_ptr<DTensor<T>> m_d_KTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_ABKTr = nullptr;  ///< (A+BK)'
    std::unique_ptr<DTensor<T>> m_d_P = nullptr;
    std::unique_ptr<DTensor<T>> m_d_APB = nullptr;  ///< (A+BK)'PB
    std::unique_ptr<DTensor<T>> m_d_Pe = nullptr;
    std::unique_ptr<DTensor<T>> m_d_lowerCholesky = nullptr;
    std::vector<std::unique_ptr<DTensor<T>>> m_choleskyStage;
    std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>> m_choleskyBatch;
    /* Workspaces */
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_d = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workX = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workU = nullptr;
    std::unique_ptr<DTensor<T>> m_d_workXU = nullptr;

    explicit Cost(ScenarioTree<T> &tree) : m_tree(tree) {
        /* Read data from files */
        m_d_BTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_BTr" + m_tree.fpFileExt()));
        m_d_KTr = std::make_unique<DTensor<T>>(m_d_K->tr());
        /* Allocate workspaces */
        m_d_q = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_tree.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_workX = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, m_tree.numNodes(), true);
        m_d_workU = std::make_unique<DTensor<T>>(m_tree.numInputs(), 1, m_tree.numNodes(), true);
        m_d_workXU = std::make_unique<DTensor<T>>(m_tree.numStatesAndInputs(), 1, m_tree.numNodes(), true);
    }

    virtual std::ostream &print(std::ostream &out) const {
        bool linear = true; //((*m_d_)(0, 0, 0) < 1e-6 && (*m_d_)(0, 0, 0) < 1e-6);
        if (linear) out << "Cost: Quadratic and linear\n";
        else out << "Cost: Quadratic\n";
        return out;
    }

public:
    virtual ~Cost() = default;

    /**
     * For reuse while testing.
     */
    void resetWorkspace() {
        m_d_q->upload(std::vector<T>(m_tree.numStates() * m_tree.numNodes(), 0));
        m_d_d->upload(std::vector<T>(m_tree.numInputs() * m_tree.numNonleafNodes(), 0));
        m_d_workX->upload(std::vector<T>(m_tree.numStates() * m_tree.numNodes(), 0));
        m_d_workU->upload(std::vector<T>(m_tree.numInputs() * m_tree.numNodes(), 0));
        m_d_workXU->upload(std::vector<T>(m_tree.numStatesAndInputs() * m_tree.numNodes(), 0));
    }

    void project(DTensor<T> &initState, DTensor<T> &states, DTensor<T> &inputs) {
        this->projectOnDynamics(initState, states, inputs);
    }

    friend std::ostream &operator<<(std::ostream &out, const Cost<T> &data) { return data.print(out); }
};


#endif
