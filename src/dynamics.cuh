#ifndef DYNAMICS_CUH
#define DYNAMICS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"


/**
 * Base class for dynamics
 */
template<typename T>
class Dynamics {

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

    explicit Dynamics(ScenarioTree<T> &tree) : m_tree(tree) {
        /* Read projection data from files */
        m_d_BTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_BTr" + m_tree.fpFileExt()));
        m_d_AB = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_AB" + m_tree.fpFileExt()));
        m_d_K = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_K" + m_tree.fpFileExt()));
        m_d_ABKTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_ABKTr" + m_tree.fpFileExt()));
        m_d_P = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_P" + m_tree.fpFileExt()));
        m_d_APB = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_APB" + m_tree.fpFileExt()));
        m_d_lowerCholesky = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynamics_lowCholesky" + m_tree.fpFileExt()));
        m_d_KTr = std::make_unique<DTensor<T>>(m_d_K->tr());
        /* Sort Cholesky data */
        m_choleskyStage = std::vector<std::unique_ptr<DTensor<T>>>(m_tree.numStagesMinus1());
        m_choleskyBatch = std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>>(m_tree.numStagesMinus1());
        for (size_t stage = 0; stage < m_tree.numStagesMinus1(); stage++) {
            size_t nodeFr = m_tree.stageFrom()[stage];
            size_t nodeTo = m_tree.stageTo()[stage];
            m_choleskyStage[stage] = std::make_unique<DTensor<T>>(*m_d_lowerCholesky, m_matAxis, nodeFr, nodeTo);
            m_choleskyBatch[stage] = std::make_unique<CholeskyBatchFactoriser<T>>(*m_choleskyStage[stage], true);
        }
        /* Allocate workspaces */
        m_d_q = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_tree.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_workX = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, m_tree.numNodes(), true);
        m_d_workU = std::make_unique<DTensor<T>>(m_tree.numInputs(), 1, m_tree.numNodes(), true);
        m_d_workXU = std::make_unique<DTensor<T>>(m_tree.numStatesAndInputs(), 1, m_tree.numNodes(), true);
    }

    void projectOnDynamics(DTensor<T> &initState, DTensor<T> &states, DTensor<T> &inputs) {
        /*
         * Set first q
         */
        DTensor<T> x_LastStage(states, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodesMinus1());
        x_LastStage *= -1.;
        memCpyNode2Node(*m_d_q, states, m_tree.numNonleafNodes(), m_tree.numNodesMinus1(), m_tree.numStates());
        /*
         * Solve for all d at current stage
         */
        size_t horizon = m_tree.numStagesMinus1();
        for (size_t t = 1; t < m_tree.numStages(); t++) {
            size_t stage = horizon - t;  // Current stage (reverse from N-1 to 0)
            size_t stageFr = m_tree.stageFrom()[stage];  // First node of current stage
            size_t stageTo = m_tree.stageTo()[stage];  // Last node of current stage
            size_t chStage = stage + 1;  // Stage of children of current stage
            size_t chStageFr = m_tree.stageFrom()[chStage];  // First node of child stage
            size_t chStageTo = m_tree.stageTo()[chStage];  // Last node of child stage
            size_t maxCh = m_tree.childMax()[stage];  // Max number of children of any node at current stage
            /* Compute Bq = B(q+Pe) at every child of current stage */
            DTensor<T> Btr_ChStage(*m_d_BTr, m_matAxis, chStageFr, chStageTo);
            DTensor<T> q_ChStage(*m_d_q, m_matAxis, chStageFr, chStageTo);
            if (m_affine) {
                DTensor<T> Pe_ChStage(*m_d_Pe, m_matAxis, chStageFr, chStageTo);
                q_ChStage += Pe_ChStage;
            }
            DTensor<T> Bq_ChStage(*m_d_workU, m_matAxis, chStageFr, chStageTo);
            Bq_ChStage.addAB(Btr_ChStage, q_ChStage);
            /* Sum `Bq` children of each node into `d` at current stage */
            for (size_t chIdx = 0; chIdx < maxCh; chIdx++) {
                m_tree.memCpyCh2Node(*m_d_d, *m_d_workU, stageFr, stageTo, chIdx, chIdx);
            }
            /* Subtract d from u in place */
            DTensor<T> d_Stage(*m_d_d, m_matAxis, stageFr, stageTo);
            DTensor<T> u_Stage(inputs, m_matAxis, stageFr, stageTo);
            d_Stage *= -1.;
            d_Stage += u_Stage;
            /* Use Cholesky decomposition for final step of computing all d at current stage */
            m_choleskyBatch[stage]->solve(d_Stage);
            /*
             * Solve for all q at current stage (except for stage 0)
             */
            if (stage) {
                /* Compute APBdAq_ChStage = A(PBd+q+Pe) for each node at child stage. A = (A+B@K).tr */
                DTensor<T> APB_ChStage(*m_d_APB, m_matAxis, chStageFr, chStageTo);
                DTensor<T> ABKtr_ChStage(*m_d_ABKTr, m_matAxis, chStageFr, chStageTo);
                m_tree.memCpyAnc2Node(*m_d_workU, *m_d_d, chStageFr, chStageTo, m_d_d->numRows());
                DTensor<T> q_SumChStage(*m_d_workX, m_matAxis, chStageFr, chStageTo);
                DTensor<T> d_ExpandedChStage(*m_d_workU, m_matAxis, chStageFr, chStageTo);
                q_SumChStage.addAB(APB_ChStage, d_ExpandedChStage);
                q_SumChStage.addAB(ABKtr_ChStage, q_ChStage, 1., 1.);
                /* Sum `APBdAq` children of each node into `q` at current stage */
                for (size_t chIdx = 0; chIdx < maxCh; chIdx++) {
                    m_tree.memCpyCh2Node(*m_d_q, *m_d_workX, stageFr, stageTo, chIdx, chIdx);
                }
                /* Compute Kdux = K.tr(d-u)@x for each node at current stage and add to `q` */
                DTensor<T> du_Stage(*m_d_workU, m_matAxis, stageFr, stageTo);
                d_Stage.deviceCopyTo(du_Stage);
                du_Stage -= u_Stage;
                DTensor<T> Ktr_Stage(*m_d_KTr, m_matAxis, stageFr, stageTo);
                DTensor<T> q_Stage(*m_d_q, m_matAxis, stageFr, stageTo);
                q_Stage.addAB(Ktr_Stage, du_Stage, 1., 1.);
                DTensor<T> x_Stage(states, m_matAxis, stageFr, stageTo);
                q_Stage -= x_Stage;
            }
        }
        /*
         * Set initial state
         */
        memCpyNode2Node(states, initState, 0, 0, m_tree.numStates());
        for (size_t stage = 0; stage < m_tree.numStagesMinus1(); stage++) {
            size_t stageFr = m_tree.stageFrom()[stage];
            size_t stageTo = m_tree.stageTo()[stage];
            size_t chStage = stage + 1;  // Stage of children of current stage
            size_t chStageFr = m_tree.stageFrom()[chStage];  // First node of child stage
            size_t chStageTo = m_tree.stageTo()[chStage];  // Last node of child stage
            /*
             * Compute next control action
             */
            DTensor<T> uAtStage(inputs, m_matAxis, stageFr, stageTo);
            DTensor<T> KAtStage(*m_d_K, m_matAxis, stageFr, stageTo);
            DTensor<T> xAtStage(states, m_matAxis, stageFr, stageTo);
            DTensor<T> dAtStage(*m_d_d, m_matAxis, stageFr, stageTo);
            uAtStage.addAB(KAtStage, xAtStage);
            uAtStage += dAtStage;
            /*
             * Compute child states
             */
            /* Fill `xu` */
            m_tree.memCpyAnc2Node(*m_d_workXU, states, chStageFr, chStageTo, m_tree.numStates());
            m_tree.memCpyAnc2Node(*m_d_workXU, inputs, chStageFr, chStageTo, m_tree.numInputs(), m_tree.numStates());
            DTensor<T> x_ChStage(states, m_matAxis, chStageFr, chStageTo);
            DTensor<T> AB_ChStage(*m_d_AB, m_matAxis, chStageFr, chStageTo);
            DTensor<T> xu_ChStage(*m_d_workXU, m_matAxis, chStageFr, chStageTo);
            x_ChStage.addAB(AB_ChStage, xu_ChStage);
            if (m_affine) {
                DTensor<T> e_ChStage(*m_d_e, m_matAxis, chStageFr, chStageTo);
                x_ChStage += e_ChStage;
            }
        }
    }

    virtual std::ostream &print(std::ostream &out) const { return out; };

public:
    virtual ~Dynamics() = default;

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

    friend std::ostream &operator<<(std::ostream &out, const Dynamics<T> &data) { return data.print(out); }
};


/**
 * Linear dynamics of the form
 * x(i+) = A(i+)x(i) + B(i+)u(i)
*/
template<typename T>
class Linear : public Dynamics<T> {

protected:
    std::ostream &print(std::ostream &out) const {
        out << "Dynamics: Linear\n";
        return out;
    }

public:
    explicit Linear(ScenarioTree<T> &tree) : Dynamics<T>(tree) {}
};


/**
 * Affine dynamics of the form
 * x(i+) = A(i+)x(i) + B(i+)u(i) + e(i+)
*/
template<typename T>
class Affine : public Dynamics<T> {

protected:
    std::ostream &print(std::ostream &out) const {
        out << "Dynamics: Affine\n";
        return out;
    }

public:
    explicit Affine(ScenarioTree<T> &tree) : Dynamics<T>(tree) {
        this->m_affine = true;
        this->m_d_e = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(this->m_tree.path() + "dynamics_e" + this->m_tree.fpFileExt()));
        this->m_d_Pe = std::make_unique<DTensor<T>>(*this->m_d_P * *this->m_d_e);
    }
};


#endif
