#ifndef PROBLEM_CUH
#define PROBLEM_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "constraints.cuh"
#include "risks.cuh"


/**
 * Store problem data:
 * - from file
 *
 * Notes:
 * - use column-major storage
 * - `d_` indicates a device pointer
 */
TEMPLATE_WITH_TYPE_T
class ProblemData {

private:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree of problem
    size_t m_numStates = 0;  ///< Total number system states
    size_t m_numInputs = 0;  ///< Total number control inputs
    size_t m_numStatesAndInputs = 0;
    size_t m_numY = 0;  ///< Size of primal vector 'y'
    T m_stepSize = 0;  ///< Step size of CP operator T
    T m_stepSizeRecip = 0;  ///< Reciprocal of step size of CP operator T
    std::unique_ptr<DTensor<T>> m_d_stepSize = nullptr;  ///< Step size of CP operator T
    std::unique_ptr<DTensor<T>> m_d_inputDynamicsTr = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateInputDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_sqrtStateWeight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtInputWeight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtStateWeightLeaf = nullptr;
    std::unique_ptr<Constraint<T>> m_nonleafConstraint;  ///< Ptr to
    std::unique_ptr<Constraint<T>> m_leafConstraint;  ///< Ptr to
    std::unique_ptr<CoherentRisk<T>> m_risk;  ///< Ptr to
    /* Dynamics projection */
    std::unique_ptr<DTensor<T>> m_d_lowerCholesky = nullptr;
    std::unique_ptr<DTensor<T>> m_d_K = nullptr;
    std::unique_ptr<DTensor<T>> m_d_KTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_dynamicsSumTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_P = nullptr;
    std::unique_ptr<DTensor<T>> m_d_APB = nullptr;
    std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>> m_choleskyBatch;
    std::vector<std::unique_ptr<DTensor<T>>> m_choleskyStage;
    /* Kernel projection */
    size_t m_nullDim = 0;  ///<
    std::unique_ptr<DTensor<T>> m_d_nullspaceProj = nullptr;
    std::unique_ptr<DTensor<T>> m_d_constraintMatrix = nullptr;
    /* Other */
    std::unique_ptr<DTensor<T>> m_d_b = nullptr;
    std::unique_ptr<DTensor<T>> m_d_bTr = nullptr;

    void parseConstraint(const rapidjson::Document &doc, std::unique_ptr<Constraint<T>> &constraint,
                         ConstraintMode mode) {
        std::string modeStr;
        size_t numNodes;
        if (mode == nonleaf) {
            modeStr = "nonleafConstraint";
            numNodes = m_tree.numNonleafNodes();
        }
        else if (mode == leaf) {
            modeStr = "leafConstraint";
            numNodes = m_tree.numLeafNodes();
        }
        std::string typeStr = doc[modeStr.c_str()].GetString();
        if (typeStr == std::string("no")) {
            constraint = std::make_unique<NoConstraint<T>>();
        } else if (typeStr == std::string("rectangle")) {
            constraint = std::make_unique<Rectangle<T>>(m_tree.path() + modeStr, m_tree.fpFileExt(),
                                                        numNodes, m_numStates, m_numInputs);
        } else if (typeStr == std::string("polyhedron")) {
            constraint = std::make_unique<Polyhedron<T>>(m_tree.path() + modeStr, m_tree.fpFileExt(), mode,
                                                         numNodes, m_numStates, m_numInputs);
        } else if (typeStr == std::string("mixed")) {
            constraint = std::make_unique<PolyhedronWithIdentity<T>>(m_tree.path() + modeStr, m_tree.fpFileExt(),
                                                                     numNodes, m_numStates, m_numInputs);
        } else {
            err << "[parseConstraint] Constraint type " << typeStr
                << " is not supported. Supported types include: none, rectangle, polyhedron, mixed" << "\n";
            throw std::invalid_argument(err.str());
        }
    }

    void parseRisk(const rapidjson::Value &value) {
        std::string typeStr = value["type"].GetString();
        if (typeStr == std::string("avar")) {
            m_risk = std::make_unique<AVaR<T>>(m_tree.path(), m_tree.fpFileExt(), m_tree.numChildren());
        } else {
            err << "[parseRisk] Risk type " << typeStr
                << " is not supported. Supported types include: avar" << "\n";
            throw std::invalid_argument(err.str());
        }
    }

    std::ostream &print(std::ostream &out) const {
        out << "Number of states: " << m_numStates << "\n";
        out << "Number of inputs: " << m_numInputs << "\n";
        out << "Nonleaf constraint: " << *m_nonleafConstraint;
        out << "Leaf constraint: " << *m_leafConstraint;
        out << "Risk: " << *m_risk;
        return out;
    }

public:
    /**
     * Constructor from JSON file stream
     */
    ProblemData(ScenarioTree<T> &tree) : m_tree(tree) {
        std::ifstream file(tree.path() + tree.json());
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        if (doc.HasParseError()) {
            err << "[Problem] Cannot parse problem data JSON file: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw std::invalid_argument(err.str());
        }

        /* Store single element data from JSON in host memory */
        m_numStates = doc["numStates"].GetInt();
        m_numInputs = doc["numInputs"].GetInt();
        m_numStatesAndInputs = m_numStates + m_numInputs;
        m_nullDim = doc["rowsNNtr"].GetInt();
        m_numY = m_nullDim - (m_tree.numEvents() * 2);
        m_stepSize = doc["stepSize"].GetDouble();
        m_stepSizeRecip = 1. / m_stepSize;

        /* Allocate memory on device */
        std::string ext = m_tree.fpFileExt();
        m_d_stepSize = std::make_unique<DTensor<T>>(std::vector(1, m_stepSize), 1);
        m_d_inputDynamicsTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "inputDynTr" + ext));
        m_d_stateInputDynamics = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "AB_dyn" + ext));
        m_d_sqrtStateWeight = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "sqrtStateCost" + ext));
        m_d_sqrtInputWeight = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "sqrtInputCost" + ext));
        m_d_sqrtStateWeightLeaf = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "sqrtTerminalCost" + ext));
        m_d_lowerCholesky = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "lowChol" + ext));
        m_d_K = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "K" + ext));
        m_d_dynamicsSumTr = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "dynTr" + ext));
        m_d_P = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "P" + ext));
        m_d_APB = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "APB" + ext));
        m_d_KTr = std::make_unique<DTensor<T>>(m_d_K->tr());

        /* Parse constraints, risks, and Cholesky data */
        parseConstraint(doc, m_nonleafConstraint, nonleaf);
        parseConstraint(doc, m_leafConstraint, leaf);
        parseRisk(doc["risk"]);
        m_choleskyBatch = std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>>(m_tree.numStagesMinus1());
        m_choleskyStage = std::vector<std::unique_ptr<DTensor<T>>>(m_tree.numStagesMinus1());
        for (size_t stage = 0; stage < m_tree.numStagesMinus1(); stage++) {
            size_t nodeFr = m_tree.stageFrom()[stage];
            size_t nodeTo = m_tree.stageTo()[stage];
            m_choleskyStage[stage] = std::make_unique<DTensor<T>>(*m_d_lowerCholesky, 2, nodeFr, nodeTo);
            m_choleskyBatch[stage] = std::make_unique<CholeskyBatchFactoriser<T>>(*m_choleskyStage[stage], true);
        }
    }

    /**
     * Destructor
     */
    ~ProblemData() = default;

    /**
     * Getters
     */
    size_t numStates() { return m_numStates; }

    size_t numInputs() { return m_numInputs; }

    size_t numStatesAndInputs() { return m_numStatesAndInputs; }

    T stepSize() { return m_stepSize; }

    T stepSizeRecip() { return m_stepSizeRecip; }

    size_t nullDim() { return m_nullDim; }

    size_t yDim() { return m_numY; }

    DTensor<T> &d_stepSize() { return *m_d_stepSize; }

    DTensor<T> &inputDynamicsTr() { return *m_d_inputDynamicsTr; }

    DTensor<T> &stateInputDynamics() { return *m_d_stateInputDynamics; }

    DTensor<T> &sqrtStateWeight() { return *m_d_sqrtStateWeight; }

    DTensor<T> &sqrtInputWeight() { return *m_d_sqrtInputWeight; }

    DTensor<T> &sqrtStateWeightLeaf() { return *m_d_sqrtStateWeightLeaf; }

    DTensor<T> &K() { return *m_d_K; }

    DTensor<T> &KTr() { return *m_d_KTr; }

    DTensor<T> &dynamicsSumTr() { return *m_d_dynamicsSumTr; }

    DTensor<T> &P() { return *m_d_P; }

    DTensor<T> &APB() { return *m_d_APB; }

    std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>> &choleskyBatch() { return m_choleskyBatch; }

    std::unique_ptr<Constraint<T>> &nonleafConstraint() { return m_nonleafConstraint; }

    std::unique_ptr<Constraint<T>> &leafConstraint() { return m_leafConstraint; }

    std::unique_ptr<CoherentRisk<T>> &risk() { return m_risk; }

    friend std::ostream &operator<<(std::ostream &out, const ProblemData<T> &data) { return data.print(out); }
};


#endif
