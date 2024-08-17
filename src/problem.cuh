#ifndef PROBLEM_CUH
#define PROBLEM_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "constraints.cuh"
#include "risks.cuh"


TEMPLATE_WITH_TYPE_T
static void parseMatrix(size_t nodeIdx, const rapidjson::Value &value, std::unique_ptr<DTensor<T>> &matrix) {
    size_t numElements = value.Capacity();
    std::vector<T> matrixData(numElements);
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        matrixData[i] = value[i].GetDouble();
    }
    DTensor<T> sliceDevice(*matrix, 2, nodeIdx, nodeIdx);
    sliceDevice.upload(matrixData, rowMajor);
}


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
    std::ifstream &m_file;
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree of problem
    size_t m_numStates = 0;  ///< Total number system states
    size_t m_numInputs = 0;  ///< Total number control inputs
    size_t m_numStatesAndInputs = 0;
    size_t m_numY = 0;  ///< Size of primal vector 'y'
    T m_stepSize = 0;  ///< Step size of CP operator T
    T m_stepSizeRecip = 0;  ///< Reciprocal of step size of CP operator T
    std::unique_ptr<DTensor<T>> m_d_alpha = nullptr;  ///< Step size of CP operator T
    std::unique_ptr<DTensor<T>> m_d_stateDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_inputDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_inputDynamicsTr = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateInputDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateWeight = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_inputWeight = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateWeightLeaf = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_sqrtStateWeight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtInputWeight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtStateWeightLeaf = nullptr;
    std::vector<std::unique_ptr<Constraint<T>>> m_nonleafConstraint;  ///< Ptr to
    std::vector<std::unique_ptr<Constraint<T>>> m_leafConstraint;  ///< Ptr to
    std::vector<std::unique_ptr<CoherentRisk<T>>> m_risk;  ///< Ptr to
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
    size_t m_nullDim = 0;  ///< Total number system states
    std::unique_ptr<DTensor<T>> m_d_nullspaceProj = nullptr;
    std::unique_ptr<DTensor<T>> m_d_constraintMatrix = nullptr;
    /* Other */
    std::unique_ptr<DTensor<T>> m_d_b = nullptr;
    std::unique_ptr<DTensor<T>> m_d_bTr = nullptr;

    static void parseConstraint(size_t nodeIdx, const rapidjson::Value &value,
                                std::vector<std::unique_ptr<Constraint<T>>> &constraint) {
        if (value["type"].GetString() == std::string("rectangle")) {
            size_t numElements = value["lb"].Capacity();
            std::vector<T> lb(numElements);
            std::vector<T> ub(numElements);
            for (rapidjson::SizeType i = 0; i < numElements; i++) {
                lb[i] = value["lb"][i].GetDouble();
                ub[i] = value["ub"][i].GetDouble();
            }
            constraint[nodeIdx] = std::make_unique<Rectangle<T>>(nodeIdx, numElements, lb, ub);
        } else {
            err << "Constraint type " << value["type"].GetString()
                << " is not supported. Supported types include: rectangle" << "\n";
            throw std::invalid_argument(err.str());
        }
    }

    void parseRisk(size_t nodeIdx, const rapidjson::Value &value) {
        if (value["type"].GetString() == std::string("avar")) {
            parseMatrix(nodeIdx, value["NNtr"], m_d_nullspaceProj);
            parseMatrix(nodeIdx, value["b"], m_d_b);
            m_risk[nodeIdx] = std::make_unique<AVaR<T>>(nodeIdx,
                                                        m_tree.numChildren()[nodeIdx],
                                                        *m_d_nullspaceProj,
                                                        *m_d_b);
        } else {
            err << "Risk type " << value["type"].GetString()
                << " is not supported. Supported types include: avar" << "\n";
            throw std::invalid_argument(err.str());
        }
    }

public:
    /**
     * Constructor from JSON file stream
     */
    ProblemData(ScenarioTree<T> &tree, std::ifstream &file) :
        m_tree(tree), m_file(file) {
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        if (doc.HasParseError()) {
            err << "[Problem] Cannot parse problem data JSON file: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw std::invalid_argument(err.str());
        }

        /** Store single element data from JSON in host memory */
        m_numStates = doc["numStates"].GetInt();
        m_numInputs = doc["numInputs"].GetInt();
        m_numStatesAndInputs = m_numStates + m_numInputs;
        m_nullDim = doc["rowsNNtr"].GetInt();
        m_numY = m_nullDim - (m_tree.numEvents() * 2);
        m_stepSize = doc["stepSize"].GetDouble();
        m_stepSizeRecip = 1 / m_stepSize;

        /** Allocate memory on host */
        m_choleskyBatch = std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>>(m_tree.numStages() - 1);
        m_choleskyStage = std::vector<std::unique_ptr<DTensor<T>>>(m_tree.numStages() - 1);
        m_nonleafConstraint = std::vector<std::unique_ptr<Constraint<T>>>(m_tree.numNonleafNodes());
        m_leafConstraint = std::vector<std::unique_ptr<Constraint<T>>>(m_tree.numLeafNodes());
        m_risk = std::vector<std::unique_ptr<CoherentRisk<T>>>(m_tree.numNonleafNodes());

        /** Allocate memory on device */
        m_d_alpha = std::make_unique<DTensor<T>>(1, 1, 1, true);
        m_d_stateDynamics = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputDynamics = std::make_unique<DTensor<T>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_inputDynamicsTr = std::make_unique<DTensor<T>>(m_numInputs, m_numStates, m_tree.numNodes(), true);
        m_d_stateInputDynamics = std::make_unique<DTensor<T>>(m_numStates, m_numStatesAndInputs, m_tree.numNodes(),
                                                              true);
        m_d_stateWeight = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputWeight = std::make_unique<DTensor<T>>(m_numInputs, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeightLeaf = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numLeafNodes(), true);
        m_d_sqrtStateWeight = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_sqrtInputWeight = std::make_unique<DTensor<T>>(m_numInputs, m_numInputs, m_tree.numNodes(), true);
        m_d_sqrtStateWeightLeaf = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numLeafNodes(), true);
        m_d_lowerCholesky = std::make_unique<DTensor<T>>(m_numInputs, m_numInputs, m_tree.numNonleafNodes(), true);
        m_d_K = std::make_unique<DTensor<T>>(m_numInputs, m_numStates, m_tree.numNonleafNodes(), true);
        m_d_KTr = std::make_unique<DTensor<T>>(m_numStates, m_numInputs, m_tree.numNonleafNodes(), true);
        m_d_dynamicsSumTr = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_P = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_APB = std::make_unique<DTensor<T>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_nullspaceProj = std::make_unique<DTensor<T>>(m_nullDim, m_nullDim, m_tree.numNonleafNodes(), true);
        m_d_b = std::make_unique<DTensor<T>>(m_numY, 1, m_tree.numNonleafNodes(), true);
        m_d_bTr = std::make_unique<DTensor<T>>(1, m_numY, m_tree.numNonleafNodes(), true);

        /** Upload to device */
        const char *nodeString = nullptr;
        for (size_t i = 0; i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["P"][nodeString], m_d_P);
        }
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["stateDynamics"][nodeString], m_d_stateDynamics);
            parseMatrix(i, doc["inputDynamics"][nodeString], m_d_inputDynamics);
            parseMatrix(i, doc["AB"][nodeString], m_d_stateInputDynamics);
            parseMatrix(i, doc["nonleafStateCosts"][nodeString], m_d_stateWeight);
            parseMatrix(i, doc["nonleafInputCosts"][nodeString], m_d_inputWeight);
            parseMatrix(i, doc["sqrtNonleafStateCosts"][nodeString], m_d_sqrtStateWeight);
            parseMatrix(i, doc["sqrtNonleafInputCosts"][nodeString], m_d_sqrtInputWeight);
            parseMatrix(i, doc["(A+B@K)t"][nodeString], m_d_dynamicsSumTr);
            parseMatrix(i, doc["At@P@B"][nodeString], m_d_APB);
        }
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseConstraint(i, doc["nonleafConstraints"][nodeString], m_nonleafConstraint);
            parseMatrix(i, doc["lowerCholesky"][nodeString], m_d_lowerCholesky);
            parseMatrix(i, doc["K"][nodeString], m_d_K);
            parseRisk(i, doc["risks"][nodeString]);
        }
        for (size_t i = m_tree.numNonleafNodes(); i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            size_t idx = i - m_tree.numNonleafNodes();
            parseMatrix(idx, doc["leafStateCosts"][nodeString], m_d_stateWeightLeaf);
            parseMatrix(idx, doc["sqrtLeafStateCosts"][nodeString], m_d_sqrtStateWeightLeaf);
            parseConstraint(idx, doc["leafConstraints"][nodeString], m_leafConstraint);
        }
        for (size_t stage = 0; stage < m_tree.numStages() - 1; stage++) {
            size_t nodeFr = m_tree.stageFrom()[stage];
            size_t nodeTo = m_tree.stageTo()[stage];
            m_choleskyStage[stage] = std::make_unique<DTensor<T>>(*m_d_lowerCholesky, 2, nodeFr, nodeTo);
            m_choleskyBatch[stage] = std::make_unique<CholeskyBatchFactoriser<T>>(*m_choleskyStage[stage], true);
        }

        /* Update remaining fields */
        m_d_alpha->upload(std::vector{m_stepSize});
        DTensor<T> BTr = m_d_inputDynamics->tr();
        BTr.deviceCopyTo(*m_d_inputDynamicsTr);
        DTensor<T> KTr = m_d_K->tr();
        KTr.deviceCopyTo(*m_d_KTr);
        DTensor<T> bTr = m_d_b->tr();
        bTr.deviceCopyTo(*m_d_bTr);
    }

    /**
     * Destructor
     */
    ~ProblemData() {}

    /**
     * Getters
     */
    std::ifstream &file() { return m_file; }

    size_t numStates() { return m_numStates; }

    size_t numInputs() { return m_numInputs; }

    size_t numStatesAndInputs() { return m_numStatesAndInputs; }

    T stepSize() { return m_stepSize; }

    T stepSizeRecip() { return m_stepSizeRecip; }

    size_t nullDim() { return m_nullDim; }

    size_t yDim() { return m_numY; }

    DTensor<T> &d_stepSize() { return *m_d_alpha; }

    DTensor<T> &stateDynamics() { return *m_d_stateDynamics; }

    DTensor<T> &inputDynamics() { return *m_d_inputDynamics; }

    DTensor<T> &inputDynamicsTr() { return *m_d_inputDynamicsTr; }

    DTensor<T> &stateInputDynamics() { return *m_d_stateInputDynamics; }

    DTensor<T> &stateWeight() { return *m_d_stateWeight; }

    DTensor<T> &inputWeight() { return *m_d_inputWeight; }

    DTensor<T> &stateWeightLeaf() { return *m_d_stateWeightLeaf; }

    DTensor<T> &sqrtStateWeight() { return *m_d_sqrtStateWeight; }

    DTensor<T> &sqrtInputWeight() { return *m_d_sqrtInputWeight; }

    DTensor<T> &sqrtStateWeightLeaf() { return *m_d_sqrtStateWeightLeaf; }

    DTensor<T> &K() { return *m_d_K; }

    DTensor<T> &KTr() { return *m_d_KTr; }

    DTensor<T> &dynamicsSumTr() { return *m_d_dynamicsSumTr; }

    DTensor<T> &P() { return *m_d_P; }

    DTensor<T> &APB() { return *m_d_APB; }

    DTensor<T> &nullspaceProj() { return *m_d_nullspaceProj; }

    DTensor<T> &b() { return *m_d_b; }

    DTensor<T> &bTr() { return *m_d_bTr; }

    std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>> &choleskyBatch() { return m_choleskyBatch; }

    std::vector<std::unique_ptr<Constraint<T>>> &nonleafConstraint() { return m_nonleafConstraint; }

    std::vector<std::unique_ptr<Constraint<T>>> &leafConstraint() { return m_leafConstraint; }

    std::vector<std::unique_ptr<CoherentRisk<T>>> &risk() { return m_risk; }

    /**
     * Debugging
     */
    void print() {
        std::cout << "Number of states: " << m_numStates << "\n";
        std::cout << "Number of inputs: " << m_numInputs << "\n";
        printIfTensor("State dynamics (from device): ", m_d_stateDynamics);
        printIfTensor("Input dynamics (from device): ", m_d_inputDynamics);
        printIfTensor("State weight (from device): ", m_d_stateWeight);
        printIfTensor("Input weight (from device): ", m_d_inputWeight);
        printIfTensor("Terminal state weight (from device): ", m_d_stateWeightLeaf);
        std::cout << "State-input constraints: \n";
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            m_nonleafConstraint[i]->print();
        }
        std::cout << "Leaf state constraints: \n";
        for (size_t i = m_tree.numNonleafNodes(); i < m_tree.numNodes(); i++) {
            m_leafConstraint[i]->print();
        }
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            m_risk[i]->print();
        }
        printIfTensor("Lower Cholesky (from device): ", m_d_lowerCholesky);
        printIfTensor("K (from device): ", m_d_K);
        printIfTensor("A + BK (from device): ", m_d_dynamicsSumTr);
        printIfTensor("P (from device): ", m_d_P);
        printIfTensor("Nullspace projection matrix (from device): ", m_d_nullspaceProj);
    }
};


#endif
