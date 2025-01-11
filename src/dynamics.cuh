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
    std::unique_ptr<DTensor<T>> m_d_inputDynamicsTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_stateInputDynamics = nullptr;
    /* Projection */
    std::unique_ptr<DTensor<T>> m_d_lowerCholesky = nullptr;
    std::unique_ptr<DTensor<T>> m_d_K = nullptr;
    std::unique_ptr<DTensor<T>> m_d_KTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dynamicsSumTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_P = nullptr;
    std::unique_ptr<DTensor<T>> m_d_APB = nullptr;
    std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>> m_choleskyBatch;
    std::vector<std::unique_ptr<DTensor<T>>> m_choleskyStage;

    explicit Dynamics(ScenarioTree<T> &tree) : m_tree(tree) {
        m_d_inputDynamicsTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "inputDynTr" + m_tree.fpFileExt()));
        m_d_stateInputDynamics = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "AB_dyn" + m_tree.fpFileExt()));
        m_d_lowerCholesky = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "lowChol" + m_tree.fpFileExt()));
        m_d_K = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "K" + m_tree.fpFileExt()));
        m_d_dynamicsSumTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynTr" + m_tree.fpFileExt()));
        m_d_P = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "P" + m_tree.fpFileExt()));
        m_d_APB = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "APB" + m_tree.fpFileExt()));
        m_d_KTr = std::make_unique<DTensor<T>>(m_d_K->tr());
        m_choleskyBatch = std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>>(m_tree.numStagesMinus1());
        m_choleskyStage = std::vector<std::unique_ptr<DTensor<T>>>(m_tree.numStagesMinus1());
        for (size_t stage = 0; stage < m_tree.numStagesMinus1(); stage++) {
            size_t nodeFr = m_tree.stageFrom()[stage];
            size_t nodeTo = m_tree.stageTo()[stage];
            m_choleskyStage[stage] = std::make_unique<DTensor<T>>(*m_d_lowerCholesky, 2, nodeFr, nodeTo);
            m_choleskyBatch[stage] = std::make_unique<CholeskyBatchFactoriser<T>>(*m_choleskyStage[stage], true);
        }
    }

    virtual std::ostream &print(std::ostream &out) const { return out; };

public:
    virtual ~Dynamics() = default;

    virtual void project(DTensor<T> &, DTensor<T> &) {};

    friend std::ostream &operator<<(std::ostream &out, const Dynamics<T> &data) { return data.print(out); }
};


/**
 * Linear dynamics of the form
 * x(t+1) = A(t+1)x(t) + B(t+1)u(t)
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

    void project(DTensor<T> &x, DTensor<T> &u) {
        /*
         * Set first q
         */
        DTensor<T> x_LastStage(x, this->m_matAxis, this->m_tree.numNonleafNodes(), this->m_tree.numNodesMinus1());
        x_LastStage *= -1.;
        DTensor<T> q_LastStage(*m_d_q, this->m_matAxis, this->m_tree.numNonleafNodes(), this->m_tree.numNodesMinus1());
        x_LastStage.deviceCopyTo(q_LastStage);
        /*
         * Solve for all d at current stage
         */
        size_t horizon = this->m_tree.numStagesMinus1();
        for (size_t t = 1; t < this->m_tree.numStages(); t++) {
            size_t stage = horizon - t;  // Current stage (reverse from N-1 to 0)
            size_t stageFr = this->m_tree.stageFrom()[stage];  // First node of current stage
            size_t stageTo = this->m_tree.stageTo()[stage];  // Last node of current stage
            size_t chStage = stage + 1;  // Stage of children of current stage
            size_t chStageFr = this->m_tree.stageFrom()[chStage];  // First node of child stage
            size_t chStageTo = this->m_tree.stageTo()[chStage];  // Last node of child stage
            size_t maxCh = this->m_tree.childMax()[stage];  // Max number of children of any node at current stage
            /* Compute `Bq` at every child of current stage */
            DTensor<T> Btr_ChStage(*this->m_d_inputDynamicsTr, this->m_matAxis, chStageFr, chStageTo);
            DTensor<T> q_ChStage(*m_d_q, this->m_matAxis, chStageFr, chStageTo);
            DTensor<T> Bq_ChStage(*m_d_workU, this->m_matAxis, chStageFr, chStageTo);
            Bq_ChStage.addAB(Btr_ChStage, q_ChStage);
            /* Sum `Bq` children of each node into `d` at current stage */
            for (size_t chIdx = 0; chIdx < maxCh; chIdx++) {
                this->m_tree.memCpyCh2Node(*m_d_d, *m_d_workU, stageFr, stageTo, chIdx, chIdx);
            }
            /* Subtract d from u in place */
            DTensor<T> d_Stage(*m_d_d, this->m_matAxis, stageFr, stageTo);
            DTensor<T> u_Stage(*m_d_u, this->m_matAxis, stageFr, stageTo);
            d_Stage *= -1.;
            d_Stage += u_Stage;
            /* Use Cholesky decomposition for final step of computing all d at current stage */
            m_data.choleskyBatch()[stage]->solve(d_Stage);
            /*
             * Solve for all q at current stage
             */
            /* Compute APBdAq_ChStage = A(PBd+q) for each node at child stage. A = (A+B@K).tr */
            DTensor<T> APB_ChStage(m_data.APB(), this->m_matAxis, chStageFr, chStageTo);
            DTensor<T> ABKtr_ChStage(m_data.dynamicsSumTr(), this->m_matAxis, chStageFr, chStageTo);
            this->m_tree.memCpyAnc2Node(*m_d_workU, *m_d_d, chStageFr, chStageTo, m_d_d->numRows(), 0, 0);
            DTensor<T> q_SumChStage(*m_d_workX, this->m_matAxis, chStageFr, chStageTo);
            DTensor<T> d_ExpandedChStage(*m_d_workU, this->m_matAxis, chStageFr, chStageTo);
            q_SumChStage.addAB(APB_ChStage, d_ExpandedChStage);
            q_SumChStage.addAB(ABKtr_ChStage, q_ChStage, 1., 1.);
            /* Sum `APBdAq` children of each node into `q` at current stage */
            for (size_t chIdx = 0; chIdx < maxCh; chIdx++) {
                this->m_tree.memCpyCh2Node(*m_d_q, *m_d_workX, stageFr, stageTo, chIdx, chIdx);
            }
            /* Compute Kdux = K.tr(d-u)@x for each node at current stage and add to `q` */
            DTensor<T> du_Stage(*m_d_workU, this->m_matAxis, stageFr, stageTo);
            d_Stage.deviceCopyTo(du_Stage);
            du_Stage -= u_Stage;
            DTensor<T> Ktr_Stage(m_data.KTr(), this->m_matAxis, stageFr, stageTo);
            DTensor<T> q_Stage(*m_d_q, this->m_matAxis, stageFr, stageTo);
            q_Stage.addAB(Ktr_Stage, du_Stage, 1., 1.);
            DTensor<T> x_Stage(*m_d_x, this->m_matAxis, stageFr, stageTo);
            q_Stage -= x_Stage;
        }
        /*
         * Set initial state
         */
        DTensor<T> firstState(*m_d_x, this->m_matAxis, 0, 0);
        m_d_initState->deviceCopyTo(firstState);
        for (size_t stage = 0; stage < this->m_tree.numStagesMinus1(); stage++) {
            size_t stageFr = this->m_tree.stageFrom()[stage];
            size_t stageTo = this->m_tree.stageTo()[stage];
            size_t chStage = stage + 1;  // Stage of children of current stage
            size_t chStageFr = this->m_tree.stageFrom()[chStage];  // First node of child stage
            size_t chStageTo = this->m_tree.stageTo()[chStage];  // Last node of child stage
            /*
             * Compute next control action
             */
            DTensor<T> uAtStage(*m_d_u, this->m_matAxis, stageFr, stageTo);
            DTensor<T> KAtStage(m_data.K(), this->m_matAxis, stageFr, stageTo);
            DTensor<T> xAtStage(*m_d_x, this->m_matAxis, stageFr, stageTo);
            DTensor<T> dAtStage(*m_d_d, this->m_matAxis, stageFr, stageTo);
            uAtStage.addAB(KAtStage, xAtStage);
            uAtStage += dAtStage;
            /*
             * Compute child states
             */
            /* Fill `xu` */
            this->m_tree.memCpyAnc2Node(*m_d_workXU, *m_d_x, chStageFr, chStageTo, m_data.numStates(), 0, 0);
            this->m_tree.memCpyAnc2Node(*m_d_workXU, *m_d_u, chStageFr, chStageTo, m_data.numInputs(), m_data.numStates(), 0);
            DTensor<T> x_ChStage(*m_d_x, this->m_matAxis, chStageFr, chStageTo);
            DTensor<T> AB_ChStage(*this->m_d_dynamicsSumTr, this->m_matAxis, chStageFr, chStageTo);
            DTensor<T> xu_ChStage(*m_d_workXU, this->m_matAxis, chStageFr, chStageTo);
            x_ChStage.addAB(AB_ChStage, xu_ChStage);
        }
    }
};


/**
 * Affine dynamics of the form
 * x(t+1) = A(t+1)x(t) + B(t+1)u(t) + e(t+1)
*/
template<typename T>
class Affine : public Dynamics<T> {

protected:
    std::unique_ptr<DTensor<T>> m_d_affineDyn = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Dynamics: Affine\n";
        return out;
    }

public:
    explicit Affine(ScenarioTree<T> &tree) : Dynamics<T>(tree) {
        m_d_affineDyn = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(this->m_tree.path() + "inputDynTr" + this->m_tree.fpFileExt()));
    }

    void project() {

    }
};


#endif
