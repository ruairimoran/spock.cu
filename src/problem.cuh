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
    std::vector<std::unique_ptr<Constraint>> m_stateConstraint{};  ///< Ptr to
    std::vector<std::unique_ptr<Constraint>> m_inputConstraint{};  ///< Ptr to
    std::vector<std::unique_ptr<Constraint>> m_stateConstraintLeaf{};  ///< Ptr to
    std::vector<std::unique_ptr<CoherentRisk>> m_risk{};  ///< Ptr to

    static void uploadJsonMat(size_t event, size_t numElements, const rapidjson::Value &json, DTensor<real_t> &slice) {
        std::vector<real_t> sliceJsonData(numElements, 0.);
        size_t matStart = event * numElements;
        for (size_t i = matStart; i < matStart + numElements; i++) {
            sliceJsonData[i] = json[i].GetDouble();
        }
        slice.upload(sliceJsonData, rowMajor);
    }

    static void parseConstraint(std::vector<real_t> &dst, const rapidjson::Value &json, size_t length) {
        if (json["type"].GetString() == std::string("rectangle")) {
            for (size_t i = 0; i < length; i++) {
                dst.push_back(json["lowerBound"][i].GetDouble());
            }
            for (size_t i = 0; i < length; i++) {
                dst.push_back(json["upperBound"][i].GetDouble());
            }
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
        size_t lenDoubleState = m_numStates * 2;
        size_t lenDoubleInput = m_numInputs * 2;

        /** Allocate memory on host for JSON data */
        std::vector<real_t> jsonStateDynamics(lenStateMat * m_tree.numEvents());
        std::vector<real_t> jsonInputDynamics(lenInputDynMat * m_tree.numEvents());
        std::vector<real_t> jsonStateWeight(lenStateMat * m_tree.numEvents());
        std::vector<real_t> jsonInputWeight(lenInputWgtMat * m_tree.numEvents());
        std::vector<real_t> jsonStateWeightLeaf(lenStateMat * m_tree.numEvents());

        /** Allocate memory on device */
        m_d_stateDynamics = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputDynamics = std::make_unique<DTensor<real_t>>(m_numStates, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeight = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);
        m_d_inputWeight = std::make_unique<DTensor<real_t>>(m_numInputs, m_numInputs, m_tree.numNodes(), true);
        m_d_stateWeightLeaf = std::make_unique<DTensor<real_t>>(m_numStates, m_numStates, m_tree.numNodes(), true);

        /** Store array data from JSON in host memory */
        for (rapidjson::SizeType i = 0; i < lenStateMat * m_tree.numEvents(); i++) {
            jsonStateDynamics[i] = doc["stateDynamics"][i].GetDouble();
            jsonStateWeight[i] = doc["stateWeight"][i].GetDouble();
            jsonStateWeightLeaf[i] = doc["stateWeightLeaf"][i].GetDouble();
        }
        for (rapidjson::SizeType i = 0; i < lenInputDynMat * m_tree.numEvents(); i++) {
            jsonInputDynamics[i] = doc["controlDynamics"][i].GetDouble();
        }
        for (rapidjson::SizeType i = 0; i < lenInputWgtMat * m_tree.numEvents(); i++) {
            jsonInputWeight[i] = doc["inputWeight"][i].GetDouble();
        }
//        parseConstraint(jsonStateConstraint, doc["stateConstraint"], m_numStates);
//        parseConstraint(jsonInputConstraint, doc["inputConstraint"], m_numInputs);
//        parseConstraint(jsonStateConstraintLeaf, doc["stateConstraintLeaf"], m_numStates);
//        parseRisk(doc["risk"], m_risk);

        /** Upload to device */
        std::vector<size_t> hostEvents(m_tree.numEvents());
        m_tree.events().download(hostEvents);
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            size_t axisMat = 2;
            DTensor<real_t> sliceStateDynamics(*m_d_stateDynamics, axisMat, i, i);
            uploadJsonMat(hostEvents[i], lenStateMat, doc["stateDynamics"], sliceStateDynamics);
//            uploadJsonMat(i, hostEvents[i], lenInputDynMat, doc["controlDynamics"], *m_d_inputDynamics);
//            pushColumnMajor(hostStateWeight, jsonStateWeight.data() + (event * lenStateMat), m_numStates, m_numStates);
//            pushColumnMajor(hostInputWeight, jsonInputWeight.data() + (event * lenInputWgtMat), m_numInputs,
//                            m_numInputs);

//            hostStateConstraint.insert(hostStateConstraint.end(),
//                                       jsonStateConstraint.begin() + (event * lenDoubleState),
//                                       jsonStateConstraint.begin() + (event * lenDoubleState + lenDoubleState));

//                m_stateConstraintCone.push_back(std::make_unique<NonnegativeOrthantCone>(m_context, lenDoubleState));
//            hostInputConstraint.insert(hostInputConstraint.end(),
//                                       jsonInputConstraint.begin() + (event * lenDoubleInput),
//                                       jsonInputConstraint.begin() + (event * lenDoubleInput + lenDoubleInput));
//                m_inputConstraintCone.push_back(std::make_unique<NonnegativeOrthantCone>(m_context, lenDoubleInput));

            if (i >= m_tree.numNonleafNodes()) {
//                hostStateWeightLeaf.insert(hostStateWeightLeaf.end(),
//                                           jsonStateWeightLeaf.begin() + (event * lenStateMat),
//                                           jsonStateWeightLeaf.begin() + (event * lenStateMat + lenStateMat));
//                hostStateConstraintLeaf.insert(hostStateConstraintLeaf.end(),
//                                               jsonStateConstraintLeaf.begin() + (event * lenDoubleState),
//                                               jsonStateConstraintLeaf.begin() +
//                                               (event * lenDoubleState + lenDoubleState));
//                    m_stateConstraintLeafCone.push_back(
//                            std::make_unique<NonnegativeOrthantCone>(m_context, lenDoubleState));
            }
        }

//        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
//            parseRisk(i, doc["risk"], m_risk);
//        }

        /** Transfer array data to device */
//        m_d_stateDynamics->upload(hostStateDynamics);
//        m_d_inputDynamics->upload(hostInputDynamics);
//        m_d_stateWeight->upload(hostStateWeight);
//        m_d_inputWeight->upload(hostInputWeight);
//        m_d_stateWeightLeaf->upload(hostStateWeightLeaf);
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

    std::vector<std::unique_ptr<Constraint>> &stateConstraint() { return m_stateConstraint; }

    std::vector<std::unique_ptr<Constraint>> &inputConstraint() { return m_inputConstraint; }

    std::vector<std::unique_ptr<Constraint>> &stateConstraintLeaf() { return m_stateConstraintLeaf; }

    std::vector<std::unique_ptr<CoherentRisk>> &risk() { return m_risk; }

    /**
     * Debugging
     */
    void print() {
        size_t len = 0;
        std::cout << "Number of states: " << m_numStates << "\n";
        std::cout << "Number of inputs: " << m_numInputs << "\n";
        printIf("State dynamics (from device): ", m_d_stateDynamics);
        printIf("Input dynamics (from device): ", m_d_inputDynamics);
        printIf("State weight (from device): ", m_d_stateWeight);
        printIf("Input weight (from device): ", m_d_inputWeight);
        printIf("Leaf state weight (from device): ", m_d_stateWeightLeaf);
//        for (size_t i = 0; i < m_tree.numNodes(); i++) {
//            m_stateConstraint[i].print();
//        }
//        for (size_t i = 0; i < m_tree.numNodes(); i++) {
//            m_stateConstraint[i].print();
//        }
//        for (size_t i = 0; i < m_tree.numNodes(); i++) {
//            m_stateConstraint[i].print();
//        }
//        for (size_t i = 0; i < m_tree.numNodes(); i++) {
//            m_risk[i].print();
//        }
    }
};

#endif
