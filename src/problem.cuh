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
class ProblemData {

private:
    ScenarioTree &m_tree;  ///< Previously created scenario tree of problem
    size_t m_numStates = 0;  ///< Total number system states
    size_t m_numInputs = 0;  ///< Total number control inputs
    std::unique_ptr<DTensor<real_t>> m_d_stateDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<real_t>> m_d_inputDynamics = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<real_t>> m_d_stateWeight = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<real_t>> m_d_inputWeight = nullptr;  ///< Ptr to
    std::unique_ptr<DTensor<real_t>> m_d_stateWeightLeaf = nullptr;  ///< Ptr to
    std::vector<std::unique_ptr<Constraint<real_t>>> m_stateConstraint;  ///< Ptr to
    std::vector<std::unique_ptr<Constraint<real_t>>> m_inputConstraint;  ///< Ptr to
    std::vector<std::unique_ptr<CoherentRisk<real_t>>> m_risk;  ///< Ptr to
    /* Dynamics projection */
    std::unique_ptr<DTensor<real_t>> m_d_lowerCholesky = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_K = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_dynamicsSum = nullptr;
    std::unique_ptr<DTensor<real_t>> m_d_P = nullptr;
    std::vector<std::unique_ptr<CholeskyBatchFactoriser<real_t>>>m_choleskyBatch;
    std::vector<std::unique_ptr<DTensor<real_t>>>m_choleskyStage;
    /* Kernel projection */
    size_t m_nullDim = 0;  ///< Total number system states
    std::unique_ptr<DTensor<real_t>> m_d_nullspace = nullptr;

    static void parseMatrix(size_t nodeIdx, const rapidjson::Value &value, std::unique_ptr<DTensor<real_t>> &matrix) {
        size_t numElements = value.Capacity();
        std::vector<real_t> matrixData(numElements);
        for (rapidjson::SizeType i = 0; i < numElements; i++) {
            matrixData[i] = value[i].GetDouble();
        }
        DTensor<real_t> sliceDevice(*matrix, 2, nodeIdx, nodeIdx);
        sliceDevice.upload(matrixData, rowMajor);
    }

    static void parseConstraint(size_t nodeIdx, const rapidjson::Value &value,
                                std::vector<std::unique_ptr<Constraint<real_t>>> &constraint) {
        if (value["type"].GetString() == std::string("rectangle")) {
            size_t numElements = value["lb"].Capacity();
            std::vector<real_t> lb(numElements);
            std::vector<real_t> ub(numElements);
            for (rapidjson::SizeType i = 0; i < numElements; i++) {
                lb[i] = value["lb"][i].GetDouble();
                ub[i] = value["ub"][i].GetDouble();
            }
            constraint[nodeIdx] = std::make_unique<Rectangle<real_t>>(nodeIdx, numElements, lb, ub);
        } else {
            std::cerr << "Constraint type " << value["type"].GetString()
                      << " is not supported. Supported types include: rectangle" << "\n";
            throw std::invalid_argument("Constraint type not supported");
        }
    }

    void parseRisk(size_t nodeIdx, const rapidjson::Value &value) {
        if (value["type"].GetString() == std::string("avar")) {
            m_risk[nodeIdx] = std::make_unique<AVaR<real_t>>(value["alpha"].GetDouble(),
                                                             nodeIdx,
                                                             m_tree.numChildren()[nodeIdx],
                                                             m_tree.d_childFrom(),
                                                             m_tree.d_childTo(),
                                                             m_tree.d_conditionalProbabilities());
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
    ProblemData(ScenarioTree &tree, std::ifstream &file) :
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
        m_choleskyBatch = std::vector<std::unique_ptr<CholeskyBatchFactoriser<real_t>>>(m_tree.numStages() - 1);
        m_choleskyStage = std::vector<std::unique_ptr<DTensor<real_t>>>(m_tree.numStages() - 1);
        m_stateConstraint = std::vector<std::unique_ptr<Constraint<real_t>>>(m_tree.numNodes());
        m_inputConstraint = std::vector<std::unique_ptr<Constraint<real_t>>>(m_tree.numNodes());
        m_risk = std::vector<std::unique_ptr<CoherentRisk<real_t>>>(m_tree.numNodes());

        /** Allocate memory on device */
        m_d_stateDynamics = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputDynamics = std::make_unique<DTensor<real_t>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeight = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputWeight = std::make_unique<DTensor<real_t>>(m_numInputs, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeightLeaf = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_lowerCholesky = std::make_unique<DTensor<real_t>>(m_numInputs, m_numInputs, m_tree.numNonleafNodes(), true);
        m_d_K = std::make_unique<DTensor<real_t>>(m_numInputs, m_numStates, m_tree.numNonleafNodes(), true);
        m_d_dynamicsSum = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_P = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_nullspace = std::make_unique<DTensor<real_t>>(m_nullDim, m_nullDim, m_tree.numNonleafNodes(), true);

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
            parseMatrix(i, doc["nonleafStateCosts"][nodeString], m_d_stateWeight);
            parseMatrix(i, doc["nonleafInputCosts"][nodeString], m_d_inputWeight);
            parseConstraint(i, doc["stateConstraints"][nodeString], m_stateConstraint);
            parseMatrix(i, doc["A+BK"][nodeString], m_d_dynamicsSum);
        }
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseConstraint(i, doc["inputConstraints"][nodeString], m_inputConstraint);
            parseRisk(i, doc["risks"][nodeString]);
            parseMatrix(i, doc["lowerCholesky"][nodeString], m_d_lowerCholesky);
            parseMatrix(i, doc["K"][nodeString], m_d_K);
            parseMatrix(i, doc["nullspace"][nodeString], m_d_nullspace);
        }
        for (size_t i = m_tree.numNonleafNodes(); i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["leafStateCosts"][nodeString], m_d_stateWeightLeaf);
        }
        for (size_t stage=0; stage<m_tree.numStages()-1; stage++) {
            size_t nodeFr = m_tree.nodeFrom()[stage];
            size_t nodeTo = m_tree.nodeTo()[stage];
            m_choleskyStage[stage] = std::make_unique<DTensor<real_t>>(*m_d_lowerCholesky, 2, nodeFr, nodeTo);
            m_choleskyBatch[stage] = std::make_unique<CholeskyBatchFactoriser<real_t>>(*m_choleskyStage[stage], true);
        }
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

    DTensor<real_t> &stateDynamics() { return *m_d_stateDynamics; }

    DTensor<real_t> &inputDynamics() { return *m_d_inputDynamics; }

    DTensor<real_t> &stateWeight() { return *m_d_stateWeight; }

    DTensor<real_t> &inputWeight() { return *m_d_inputWeight; }

    DTensor<real_t> &stateWeightLeaf() { return *m_d_stateWeightLeaf; }

    DTensor<real_t> &K() { return *m_d_K; }

    DTensor<real_t> &dynamicsSum() { return *m_d_dynamicsSum; }

    DTensor<real_t> &P() { return *m_d_P; }

    DTensor<real_t> &nullspace() { return *m_d_nullspace; }

    std::vector<std::unique_ptr<CholeskyBatchFactoriser<real_t>>> &choleskyBatch() { return m_choleskyBatch; }

    std::vector<std::unique_ptr<Constraint<real_t>>> &stateConstraint() { return m_stateConstraint; }

    std::vector<std::unique_ptr<Constraint<real_t>>> &inputConstraint() { return m_inputConstraint; }

    std::vector<std::unique_ptr<CoherentRisk<real_t>>> &risk() { return m_risk; }

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
        printIfTensor("A + BK (from device): ", m_d_dynamicsSum);
        printIfTensor("P (from device): ", m_d_P);
        printIfTensor("Nullspace (from device): ", m_d_nullspace);
    }
};


#endif
