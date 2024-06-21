#ifndef CACHE_CUH
#define CACHE_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include "cones.cuh"
#include "risks.cuh"


template<typename T>
__global__ void k_setToZero(T *vec, size_t n);


/**
 * Cache of methods for proximal algorithms
 *
 * Note: `d_` indicates a device pointer
 */
TEMPLATE_WITH_TYPE_T
class Cache {

private:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
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
    std::unique_ptr<DTensor<T>> m_d_cacheError = nullptr;
    /* Other */
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_d = nullptr;
    std::unique_ptr<DTensor<T>> m_d_stateSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_inputSizeWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_stateInputSizeWorkspace = nullptr;

    /**
     * Private methods
     */
    void reshapePrimal();

public:
    /**
     * Constructor
     */
    Cache(ScenarioTree<T> &tree, ProblemData<T> &data, T tol, size_t maxIters) :
        m_tree(tree), m_data(data), m_tol(tol), m_maxIters(maxIters) {
        /* Sizes */
        m_sizeU = m_tree.numNonleafNodes() * m_data.numInputs();  ///< Inputs of all nonleaf nodes
        m_sizeX = m_tree.numNodes() * m_data.numStates();  ///< States of all nodes
        m_sizeY = m_tree.numNonleafNodes() * m_tree.numEvents();  ///< Y for all nonleaf nodes
        m_sizeT = m_tree.numNodes();  ///< T for all child nodes
        m_sizeS = m_tree.numNodes();  ///< S for all child nodes
        m_primSize = m_sizeU + m_sizeX + m_sizeY + m_sizeT + m_sizeS;
        /* Allocate memory on device */
        m_d_prim = std::make_unique<DTensor<T>>(m_primSize, true);
        m_d_primPrev = std::make_unique<DTensor<T>>(m_primSize, true);
        m_d_dual = std::make_unique<DTensor<T>>(m_dualSize, true);
        m_d_dualPrev = std::make_unique<DTensor<T>>(m_dualSize, true);
        m_d_cacheError = std::make_unique<DTensor<T>>(m_maxIters, true);
        m_d_q = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_d = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_stateSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_inputSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        m_d_stateInputSizeWorkspace = std::make_unique<DTensor<T>>(m_data.numStates() + m_data.numInputs(), 1, m_tree.numNodes(), true);
        /* Slice primal */
        reshapePrimal();
    }

    ~Cache() {}

    /**
     * Public methods
     */
    void initialiseState(std::vector<T> &initState);

    void projectOnDynamics();

    void projectOnKernel();

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
    m_d_y->reshape(m_tree.numEvents(), 1, m_tree.numNonleafNodes());
    start += m_sizeY;
    m_d_t = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeT - 1);
    m_d_t->reshape(1, 1, m_tree.numNodes());
    start += m_sizeT;
    m_d_s = std::make_unique<DTensor<T>>(*m_d_prim, rowAxis, start, start + m_sizeS - 1);
    m_d_s->reshape(1, 1, m_tree.numNodes());
}

template<typename T>
void Cache<T>::initialiseState(std::vector<T> &initState) {
    /* Set initial state */
    if (initState.size() != m_data.numStates()) {
        std::cerr << "Error initialising state: problem setup for " << m_data.numStates()
                  << " but given " << initState.size() << " states" << "\n";
        throw std::invalid_argument("Incorrect dimension of initial state");
    }
    DTensor<T> firstState(*m_d_x, m_matAxis, 0, 0);
    firstState.upload(initState);
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
        DTensor<T> Bq_ChStage(*m_d_inputSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        Bq_ChStage.addAB(Btr_ChStage, q_ChStage);
        /* Copy into `d` the first child `Bq` of each node (every nonleaf node has at least one child) */
        for (size_t node = stageFr; node <= stageTo; node++) {
            size_t ch = m_tree.childFrom()[node];
            DTensor<T> Bq_ChNode(*m_d_inputSizeWorkspace, m_matAxis, ch, ch);
            DTensor<T> Bq_Node(*m_d_d, m_matAxis, node, node);
            Bq_ChNode.deviceCopyTo(Bq_Node);
        }
        /* Add to `d` remaining child `Bq` of each node (zeros added if child does not exist) */
        DTensor<T> d_Stage(*m_d_d, m_matAxis, stageFr, stageTo);
        for (size_t chIdx = 1; chIdx < maxCh; chIdx++) {  // Index of child of every node at current stage
            for (size_t node = stageFr; node <= stageTo; node++) {
                size_t ch = m_tree.childFrom()[node] + chIdx;
                if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Bq_ChStage`
                    DTensor<T> Bq_ChNode(*m_d_inputSizeWorkspace, m_matAxis, ch, ch);
                    DTensor<T> Bq_Node(*m_d_inputSizeWorkspace, m_matAxis, node, node);
                    Bq_ChNode.deviceCopyTo(Bq_Node);
                }
            }
            DTensor<T> d_Add(*m_d_inputSizeWorkspace, m_matAxis, stageFr, stageTo);
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
                DTensor<T> d_ExpandedChNode(*m_d_inputSizeWorkspace, m_matAxis, ch, ch);
                d_Node.deviceCopyTo(d_ExpandedChNode);
            }
        }
        DTensor<T> q_SumChStage(*m_d_stateSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        DTensor<T> d_ExpandedChStage(*m_d_inputSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        q_SumChStage.addAB(APB_ChStage, d_ExpandedChStage);
        q_SumChStage.addAB(ABKtr_ChStage, q_ChStage, 1., 1.);
        /* Copy into `q` the first child `APBdAq` of each node (every nonleaf node has at least one child) */
        for (size_t node = stageFr; node <= stageTo; node++) {
            size_t ch = m_tree.childFrom()[node];
            DTensor<T> APBdAq_ChNode(*m_d_stateSizeWorkspace, m_matAxis, ch, ch);
            DTensor<T> APBdAq_Node(*m_d_q, m_matAxis, node, node);
            APBdAq_ChNode.deviceCopyTo(APBdAq_Node);
        }
        /* Add to `q` remaining child `APBdAq` of each node (zeros added if child does not exist) */
        DTensor<T> q_Stage(*m_d_q, m_matAxis, stageFr, stageTo);
        for (size_t chOfEachNode = 1; chOfEachNode < maxCh; chOfEachNode++) {
            for (size_t node = stageFr; node <= stageTo; node++) {
                size_t ch = m_tree.childFrom()[node] + chOfEachNode;
                if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Bq_ChStage`
                    DTensor<T> APBdAq_ChNode(*m_d_stateSizeWorkspace, m_matAxis, ch, ch);
                    DTensor<T> APBdAq_Node(*m_d_stateSizeWorkspace, m_matAxis, node, node);
                    APBdAq_ChNode.deviceCopyTo(APBdAq_Node);
                }
            }
            DTensor<T> q_Add(*m_d_stateSizeWorkspace, m_matAxis, stageFr, stageTo);
            q_Stage += q_Add;
        }
        /* Compute Kdux = K.tr(d-u)@x for each node at current stage and add to `q` */
        DTensor<T> du_Stage(*m_d_inputSizeWorkspace, m_matAxis, stageFr, stageTo);
        d_Stage.deviceCopyTo(du_Stage);
        du_Stage -= u_Stage;
        DTensor<T> Ktr_Stage(m_data.KTr(), m_matAxis, stageFr, stageTo);
        q_Stage.addAB(Ktr_Stage, du_Stage, 1., 1.);
        DTensor<T> x_Stage(*m_d_x, m_matAxis, stageFr, stageTo);
        q_Stage -= x_Stage;
    }
    /*
     * Initial state has already been set, move on
     */
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
                DTensor<T> xu_ChNode(*m_d_stateInputSizeWorkspace, m_matAxis, ch, ch);
                DTensor<T> xu_sliceX(xu_ChNode, 0, 0, m_data.numStates() - 1);
                DTensor<T> xu_sliceU(xu_ChNode, 0, m_data.numStates(), m_data.numStates() + m_data.numInputs() - 1);
                x_Node.deviceCopyTo(xu_sliceX);
                u_Node.deviceCopyTo(xu_sliceU);
            }
        }
        DTensor<T> x_ChStage(*m_d_x, m_matAxis, chStageFr, chStageTo);
        DTensor<T> AB_ChStage(m_data.stateInputDynamics(), m_matAxis, chStageFr, chStageTo);
        DTensor<T> xu_ChStage(*m_d_stateInputSizeWorkspace, m_matAxis, chStageFr, chStageTo);
        x_ChStage.addAB(AB_ChStage, xu_ChStage);
    }
}

template<typename T>
void Cache<T>::projectOnKernel() {

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

/**
 * Compute one (1) iteration of vanilla CP algorithm, nothing more.
 */
template<typename T>
void Cache<T>::cpIter() {
    projectOnDynamics();
    projectOnKernel();
    /** update z_bar */
    /** update n_bar */
    /** update z */
    /** update n */
}

template<typename T>
void Cache<T>::print() {
    std::cout << "Tolerance: " << m_tol << "\n";
    std::cout << "Num iterations: " << m_countIterations << " of " << m_maxIters << "\n";
    std::cout << "Primal (from device): " << m_d_prim->tr();
}


#endif
