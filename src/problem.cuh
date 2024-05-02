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
    std::vector<std::unique_ptr<Constraint<real_t>>> m_stateConstraintLeaf;  ///< Ptr to
//    std::vector<std::unique_ptr<CoherentRisk>> m_risk;  ///< Ptr to

    void parseMatrix(size_t node_idx, size_t event, size_t numElements, const rapidjson::Value &json,
                     std::unique_ptr<DTensor<real_t>> &matrix) {
        DTensor<real_t> sliceDevice(*matrix, 2, node_idx, node_idx);
        std::vector<real_t> sliceJsonData(numElements);
        size_t start = event * numElements;
        for (size_t i = 0; i < numElements; i++) {
            sliceJsonData[i] = json[i + start].GetDouble();
        }
        sliceDevice.upload(sliceJsonData, rowMajor);
    }

    static void parseConstraint(size_t node_idx, size_t event, size_t numElements, const rapidjson::Value &json,
                                std::vector<std::unique_ptr<Constraint<real_t>>> &constraint) {
        if (json["type"][event].GetString() == std::string("rectangle")) {
            std::vector<real_t> lb(numElements);
            std::vector<real_t> ub(numElements);
            size_t start = event * numElements;
            for (size_t i = 0; i < numElements; i++) {
                lb[i] = json["lowerBound"][i + start].GetDouble();
                ub[i] = json["upperBound"][i + start].GetDouble();
            }
            constraint[node_idx] = std::make_unique<Rectangle<real_t>>(node_idx, numElements, lb, ub);
        } else {
            std::cerr << "Constraint type " << json["type"].GetString()
                      << " is not supported. Supported types include: rectangle" << "\n";
            throw std::invalid_argument("Constraint type not supported");
        }
    }

//    void parseRisk(size_t node, const rapidjson::Value &json, std::vector<std::unique_ptr<CoherentRisk>> &risk) {
//        if (json["type"].GetString() == std::string("avar")) {
//            risk[node] = std::make_unique<AVaR>(node,
//                                                json["alpha"],
//                                                m_tree.numChildren(),
//                                                m_tree.childFrom(),
//                                                m_tree.conditionalProbabilities());
//        } else {
//            std::cerr << "Risk type " << json["type"].GetString()
//                      << " is not supported. Supported types include: avar" << "\n";
//            throw std::invalid_argument("Risk type not supported");
//        }
//    }

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

        /** Sizes */
        size_t lenStateMat = m_numStates * m_numStates;
        size_t lenInputDynMat = m_numStates * m_numInputs;
        size_t lenInputWgtMat = m_numInputs * m_numInputs;

        /** Allocate memory on host */
        std::vector<real_t> jsonStateDynamics(lenStateMat * m_tree.numEvents());
        std::vector<real_t> jsonInputDynamics(lenInputDynMat * m_tree.numEvents());
        std::vector<real_t> jsonStateWeight(lenStateMat * m_tree.numEvents());
        std::vector<real_t> jsonInputWeight(lenInputWgtMat * m_tree.numEvents());
        std::vector<real_t> jsonStateWeightLeaf(lenStateMat * m_tree.numEvents());
        m_stateConstraint = std::vector<std::unique_ptr<Constraint<real_t>>>(m_tree.numNodes());
        m_inputConstraint = std::vector<std::unique_ptr<Constraint<real_t>>>(m_tree.numNodes());
        m_stateConstraintLeaf = std::vector<std::unique_ptr<Constraint<real_t>>>(m_tree.numNodes());

        /** Allocate memory on device */
        m_d_stateDynamics = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputDynamics = std::make_unique<DTensor<real_t>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeight = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputWeight = std::make_unique<DTensor<real_t>>(m_numInputs, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeightLeaf = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);

        /** Upload to device */
        std::vector<size_t> hostEvents(m_tree.numEvents());
        m_tree.events().download(hostEvents);
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            size_t event = hostEvents[i];
            parseMatrix(i, event, lenStateMat, doc["stateDynamics"], m_d_stateDynamics);
            parseMatrix(i, event, lenInputDynMat, doc["controlDynamics"], m_d_inputDynamics);
            parseMatrix(i, event, lenStateMat, doc["stateWeight"], m_d_stateWeight);
            parseMatrix(i, event, lenInputWgtMat, doc["inputWeight"], m_d_inputWeight);
            parseConstraint(i, event, m_numStates, doc["stateConstraint"], m_stateConstraint);
            parseConstraint(i, event, m_numInputs, doc["inputConstraint"], m_inputConstraint);
            if (i >= m_tree.numNonleafNodes()) {
                parseMatrix(i, event, lenStateMat, doc["stateWeightLeaf"], m_d_stateWeightLeaf);
                parseConstraint(i, event, m_numStates, doc["stateConstraintLeaf"], m_stateConstraintLeaf);
            }
        }
//        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
//            parseRisk(i, doc["risk"], m_risk);
//        }
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

    std::vector<std::unique_ptr<Constraint<real_t>>> &stateConstraint() { return m_stateConstraint; }

    std::vector<std::unique_ptr<Constraint<real_t>>> &inputConstraint() { return m_inputConstraint; }

    std::vector<std::unique_ptr<Constraint<real_t>>> &stateConstraintLeaf() { return m_stateConstraintLeaf; }

//    std::vector<std::unique_ptr<CoherentRisk>> &risk() { return m_risk; }

    /**
     * Debugging
     */
    void print() {
        std::cout << "Number of states: " << m_numStates << "\n";
        std::cout << "Number of inputs: " << m_numInputs << "\n";
        printIf("State dynamics (from device): ", m_d_stateDynamics);
        printIf("Input dynamics (from device): ", m_d_inputDynamics);
        printIf("State weight (from device): ", m_d_stateWeight);
        printIf("Input weight (from device): ", m_d_inputWeight);
        printIf("Leaf state weight (from device): ", m_d_stateWeightLeaf);
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            m_stateConstraint[i]->print();
        }
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            m_stateConstraint[i]->print();
        }
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            m_stateConstraint[i]->print();
        }
//        for (size_t i = 0; i < m_tree.numNodes(); i++) {
//            m_risk[i].print();
//        }
    }
};

#endif
