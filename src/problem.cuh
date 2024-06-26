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
    std::unique_ptr<DTensor<T>> m_d_stateDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_inputDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_inputDynamicsTr = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateInputDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateWeight = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_inputWeight = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<T>> m_d_stateWeightLeaf = nullptr;  ///< Ptr to
    std::vector<std::unique_ptr<Constraint<T>>> m_stateConstraint;  ///< Ptr to
    std::vector<std::unique_ptr<Constraint<T>>> m_inputConstraint;  ///< Ptr to
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

    static void parseMatrix(size_t nodeIdx, const rapidjson::Value &value, std::unique_ptr<DTensor<T>> &matrix) {
        size_t numElements = value.Capacity();
        std::vector<T> matrixData(numElements);
        for (rapidjson::SizeType i = 0; i < numElements; i++) {
            matrixData[i] = value[i].GetDouble();
        }
        DTensor<T> sliceDevice(*matrix, 2, nodeIdx, nodeIdx);
        sliceDevice.upload(matrixData, rowMajor);
    }

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
            std::cerr << "Constraint type " << value["type"].GetString()
                      << " is not supported. Supported types include: rectangle" << "\n";
            throw std::invalid_argument("Constraint type not supported");
        }
    }

    void parseRisk(size_t nodeIdx, const rapidjson::Value &value) {
        if (value["type"].GetString() == std::string("avar")) {
            parseMatrix(nodeIdx, value["NNtr"], m_d_nullspaceProj);
            m_risk[nodeIdx] = std::make_unique<AVaR<T>>(nodeIdx,
                                                        m_tree.numChildren()[nodeIdx],
                                                        *m_d_nullspaceProj);
        } else {
            std::cerr << "Risk type " << value["type"].GetString()
                      << " is not supported. Supported types include: avar" << "\n";
            throw std::invalid_argument("Risk type not supported");
        }
    }

public:
    /**
     * Constructor from JSON file stream
     */
    ProblemData(ScenarioTree<T> &tree, std::ifstream &file) :
        m_tree(tree) {
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());

        if (doc.HasParseError()) {
            std::cerr << "Error parsing problem data JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw std::invalid_argument("Cannot parse problem data JSON file");
        }

        /** Store single element data from JSON in host memory */
        m_numStates = doc["numStates"].GetInt();
        m_numInputs = doc["numInputs"].GetInt();
        m_nullDim = doc["nullspaceDimension"].GetInt();

        /** Allocate memory on host */
        m_choleskyBatch = std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>>(m_tree.numStages() - 1);
        m_choleskyStage = std::vector<std::unique_ptr<DTensor<T>>>(m_tree.numStages() - 1);
        m_stateConstraint = std::vector<std::unique_ptr<Constraint<T>>>(m_tree.numNodes());
        m_inputConstraint = std::vector<std::unique_ptr<Constraint<T>>>(m_tree.numNodes());
        m_risk = std::vector<std::unique_ptr<CoherentRisk<T>>>(m_tree.numNodes());

        /** Allocate memory on device */
        m_d_stateDynamics = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputDynamics = std::make_unique<DTensor<T>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_inputDynamicsTr = std::make_unique<DTensor<T>>(m_numInputs, m_numStates, m_tree.numNodes(), true);
        m_d_stateInputDynamics = std::make_unique<DTensor<T>>(m_numStates, m_numStates + m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeight = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputWeight = std::make_unique<DTensor<T>>(m_numInputs, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeightLeaf = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_lowerCholesky = std::make_unique<DTensor<T>>(m_numInputs, m_numInputs, m_tree.numNonleafNodes(), true);
        m_d_K = std::make_unique<DTensor<T>>(m_numInputs, m_numStates, m_tree.numNonleafNodes(), true);
        m_d_KTr = std::make_unique<DTensor<T>>(m_numStates, m_numInputs, m_tree.numNonleafNodes(), true);
        m_d_dynamicsSumTr = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_P = std::make_unique<DTensor<T>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_APB = std::make_unique<DTensor<T>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_nullspaceProj = std::make_unique<DTensor<T>>(m_nullDim, m_nullDim, m_tree.numNonleafNodes(), true);

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
            parseConstraint(i, doc["stateConstraints"][nodeString], m_stateConstraint);
            parseMatrix(i, doc["(A+B@K)t"][nodeString], m_d_dynamicsSumTr);
            parseMatrix(i, doc["At@P@B"][nodeString], m_d_APB);
        }
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseConstraint(i, doc["inputConstraints"][nodeString], m_inputConstraint);
            parseMatrix(i, doc["lowerCholesky"][nodeString], m_d_lowerCholesky);
            parseMatrix(i, doc["K"][nodeString], m_d_K);
            parseRisk(i, doc["risks"][nodeString]);
        }
        for (size_t i = m_tree.numNonleafNodes(); i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["leafStateCosts"][nodeString], m_d_stateWeightLeaf);
        }
        for (size_t stage = 0; stage < m_tree.numStages() - 1; stage++) {
            size_t nodeFr = m_tree.stageFrom()[stage];
            size_t nodeTo = m_tree.stageTo()[stage];
            m_choleskyStage[stage] = std::make_unique<DTensor<T>>(*m_d_lowerCholesky, 2, nodeFr, nodeTo);
            m_choleskyBatch[stage] = std::make_unique<CholeskyBatchFactoriser<T>>(*m_choleskyStage[stage], true);
        }

        /* Update remaining fields */
        DTensor<T> BTr = m_d_inputDynamics->tr();
        BTr.deviceCopyTo(*m_d_inputDynamicsTr);
        DTensor<T> KTr = m_d_K->tr();
        KTr.deviceCopyTo(*m_d_KTr);
    }

    /**
     * Destructor
     */
    ~ProblemData() {}

    /**
     * Getters
     */
    size_t numStates() { return m_numStates; }

    size_t numInputs() { return m_numInputs; }

    size_t nullDim() { return m_nullDim; }

    DTensor<T> &stateDynamics() { return *m_d_stateDynamics; }

    DTensor<T> &inputDynamics() { return *m_d_inputDynamics; }

    DTensor<T> &inputDynamicsTr() { return *m_d_inputDynamicsTr; }

    DTensor<T> &stateInputDynamics() { return *m_d_stateInputDynamics; }

    DTensor<T> &stateWeight() { return *m_d_stateWeight; }

    DTensor<T> &inputWeight() { return *m_d_inputWeight; }

    DTensor<T> &stateWeightLeaf() { return *m_d_stateWeightLeaf; }

    DTensor<T> &K() { return *m_d_K; }

    DTensor<T> &KTr() { return *m_d_KTr; }

    DTensor<T> &dynamicsSumTr() { return *m_d_dynamicsSumTr; }

    DTensor<T> &P() { return *m_d_P; }

    DTensor<T> &APB() { return *m_d_APB; }

    DTensor<T> &nullspaceProj() { return *m_d_nullspaceProj; }

    std::vector<std::unique_ptr<CholeskyBatchFactoriser<T>>> &choleskyBatch() { return m_choleskyBatch; }

    std::vector<std::unique_ptr<Constraint<T>>> &stateConstraint() { return m_stateConstraint; }

    std::vector<std::unique_ptr<Constraint<T>>> &inputConstraint() { return m_inputConstraint; }

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
        std::cout << "State constraints: \n";
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            m_stateConstraint[i]->print();
        }
        std::cout << "Input constraints: \n";
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            m_inputConstraint[i]->print();
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
