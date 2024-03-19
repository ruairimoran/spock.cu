#ifndef __PROBLEM__
#define __PROBLEM__
#include "../include/stdgpu.h"
#include "tree.cuh"
#include "constraints.cuh"
#include "risks.cuh"

static void pushColumnMajor(std::vector<real_t>& dst, const real_t* src, size_t rows, size_t cols) {
    dst.reserve(dst.size() + rows * cols);
    for (size_t c = 0; c < cols; c++) {
        for (size_t r = 0; r < rows; r++) {
            dst.push_back(src[r * cols + c]);
        }
    }
}

static void parseConstraint(std::vector<real_t>& dst, const rapidjson::Value& value, size_t length) {
    if (value["type"].GetString() != std::string("rectangle")) {
        throw std::runtime_error("invalid constraint type");
    }
    for (size_t i = 0; i < length; i++) {
        dst.push_back(value["lowerBound"][i].GetDouble());
    }
    for (size_t i = 0; i < length; i++) {
        dst.push_back(value["upperBound"][i].GetDouble());
    }
}

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
        ScenarioTree& m_tree;  ///< Previously created scenario tree of problem
        Context m_context;  ///< Handle for cublas
        size_t m_numStates = 0;  ///< Total number system states
        size_t m_numInputs = 0;  ///< Total number control inputs
        DeviceVector<real_t> m_d_stateDynamics;  ///< Ptr to
        DeviceVector<real_t> m_d_inputDynamics;  ///< Ptr to
        DeviceVector<real_t> m_d_stateWeight;  ///< Ptr to
        DeviceVector<real_t> m_d_inputWeight;  ///< Ptr to
        DeviceVector<real_t> m_d_stateWeightLeaf;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraint;  ///< Ptr to
        std::vector<std::unique_ptr<ConvexCone>> m_stateConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_inputConstraint;  ///< Ptr to
        std::vector<std::unique_ptr<ConvexCone>> m_inputConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraintLeaf;  ///< Ptr to
        std::vector<std::unique_ptr<ConvexCone>> m_stateConstraintLeafCone;  ///< Ptr to
        std::vector<std::unique_ptr<CoherentRisk>> m_risk;  ///< Ptr to

	public:
		/**
		 * Constructor from JSON file stream
		 */
		ProblemData(ScenarioTree& tree, std::ifstream& file) : m_tree(tree) {
            std::string json((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            rapidjson::Document doc;
            doc.Parse(json.c_str());

            if (doc.HasParseError()) {
              std::cerr << "Error parsing problem data JSON: " << GetParseError_En(doc.GetParseError()) << std::endl;
              throw std::invalid_argument("Cannot parse problem data JSON file");
            }

            /** Store single element data from JSON in host memory */
            m_numStates = doc["numStates"].GetInt();
            m_numInputs = doc["numInputs"].GetInt();

            /** Sizes */
            size_t lenStateMat = m_numStates * m_numStates;
            size_t lenInputDynMat = m_numInputs * m_numStates;
            size_t lenInputWgtMat = m_numInputs * m_numInputs;
            size_t lenDoubleState = m_numStates * 2;
            size_t lenDoubleInput = m_numInputs * 2;

            /** Allocate memory on host for JSON data */
            std::vector<real_t> jsonStateDynamics(lenStateMat * m_tree.numEvents());
            std::vector<real_t> jsonInputDynamics(lenInputDynMat * m_tree.numEvents());
            std::vector<real_t> jsonStateWeight(lenStateMat * m_tree.numEvents());
            std::vector<real_t> jsonInputWeight(lenInputWgtMat * m_tree.numEvents());
            std::vector<real_t> jsonStateWeightLeaf(lenStateMat * m_tree.numEvents());
            std::vector<real_t> jsonStateConstraint{};
            std::vector<real_t> jsonInputConstraint{};
            std::vector<real_t> jsonStateConstraintLeaf{};
            std::string jsonRiskType;
            real_t jsonRiskAlpha;

            /** Allocate memory on device */
            m_d_stateDynamics.allocateOnDevice(lenStateMat * m_tree.numNodes());
            m_d_inputDynamics.allocateOnDevice(lenInputDynMat * m_tree.numNodes());
            m_d_stateWeight.allocateOnDevice(lenStateMat * m_tree.numNodes());
            m_d_inputWeight.allocateOnDevice(lenInputWgtMat * m_tree.numNodes());
            m_d_stateWeightLeaf.allocateOnDevice(lenStateMat * m_tree.numNodes());
            m_d_stateConstraint.allocateOnDevice(lenDoubleState * m_tree.numNodes());
            m_d_inputConstraint.allocateOnDevice(lenDoubleInput * m_tree.numNodes());
            m_d_stateConstraintLeaf.allocateOnDevice(lenDoubleState * m_tree.numNodes());

            /** Store array data from JSON in host memory */
            for (rapidjson::SizeType i = 0; i<lenStateMat; i++) {
                jsonStateDynamics[i] = doc["stateDynamicsMode0"][i].GetDouble();
                jsonStateDynamics[i+lenStateMat] = doc["stateDynamicsMode1"][i].GetDouble();
                jsonStateWeight[i] = doc["stateWeightMode0"][i].GetDouble();
                jsonStateWeight[i+lenStateMat] = doc["stateWeightMode1"][i].GetDouble();
                jsonStateWeightLeaf[i] = doc["stateWeightLeafMode0"][i].GetDouble();
                jsonStateWeightLeaf[i+lenStateMat] = doc["stateWeightLeafMode1"][i].GetDouble();
            }

            for (rapidjson::SizeType i = 0; i<lenInputDynMat; i++) {
                jsonInputDynamics[i] = doc["controlDynamicsMode0"][i].GetDouble();
                jsonInputDynamics[i+lenInputDynMat] = doc["controlDynamicsMode1"][i].GetDouble();
            }

            for (rapidjson::SizeType i = 0; i<lenInputWgtMat; i++) {
                jsonInputWeight[i] = doc["inputWeightMode0"][i].GetDouble();
                jsonInputWeight[i+lenInputWgtMat] = doc["inputWeightMode1"][i].GetDouble();
            }

            parseConstraint(jsonStateConstraint, doc["stateConstraintMode0"], m_numStates);
            parseConstraint(jsonStateConstraint, doc["stateConstraintMode1"], m_numStates);
            parseConstraint(jsonStateConstraintLeaf, doc["stateConstraintLeafMode0"], m_numStates);
            parseConstraint(jsonStateConstraintLeaf, doc["stateConstraintLeafMode1"], m_numStates);
            parseConstraint(jsonInputConstraint, doc["inputConstraintMode0"], m_numInputs);
            parseConstraint(jsonInputConstraint, doc["inputConstraintMode1"], m_numInputs);

            jsonRiskType = doc["risk"]["type"].GetString();
            jsonRiskAlpha = doc["risk"]["alpha"].GetDouble();

            /** Create full arrays on host */
            std::vector<real_t> hostStateDynamics(lenStateMat, 0.);
            std::vector<real_t> hostInputDynamics(lenInputDynMat, 0.);
            std::vector<real_t> hostStateWeight(lenStateMat, 0.);
            std::vector<real_t> hostInputWeight(lenInputWgtMat, 0.);
            std::vector<real_t> hostStateWeightLeaf(lenStateMat * m_tree.numNonleafNodes(), 0.);

            std::vector<real_t> hostStateConstraint(lenDoubleState, 0.);
            m_stateConstraintCone.push_back(std::make_unique<NullCone>(m_context, 0));
            
            std::vector<real_t> hostInputConstraint(lenDoubleInput, 0.);
            m_inputConstraintCone.push_back(std::make_unique<NullCone>(m_context, 0));
            
            std::vector<real_t> hostStateConstraintLeaf(lenDoubleState * m_tree.numNonleafNodes(), 0.);
            for (size_t i=0; i<m_tree.numNonleafNodes(); i++) m_stateConstraintLeafCone.push_back(std::make_unique<NullCone>(m_context, 0));

            std::vector<size_t> hostEvents(m_tree.events().capacity());
            m_tree.events().download(hostEvents);

            for (size_t i=1; i<m_tree.numNodes(); i++) {
                size_t event = hostEvents[i];
                pushColumnMajor(hostStateDynamics, jsonStateDynamics.data() + (event * lenStateMat), m_numStates, m_numStates);
                pushColumnMajor(hostInputDynamics, jsonInputDynamics.data() + (event * lenInputDynMat), m_numStates, m_numInputs);
                pushColumnMajor(hostStateWeight, jsonStateWeight.data() + (event * lenStateMat), m_numStates, m_numStates);
                pushColumnMajor(hostInputWeight, jsonInputWeight.data() + (event * lenInputWgtMat), m_numInputs, m_numInputs);

                hostStateConstraint.insert(hostStateConstraint.end(),
                                           jsonStateConstraint.begin() + (event * lenDoubleState),
                                           jsonStateConstraint.begin() + (event * lenDoubleState + lenDoubleState));

                m_stateConstraintCone.push_back(std::make_unique<NonnegativeOrthantCone>(m_context, lenDoubleState));
                hostInputConstraint.insert(hostInputConstraint.end(),
                                           jsonInputConstraint.begin() + (event * lenDoubleInput),
                                           jsonInputConstraint.begin() + (event * lenDoubleInput + lenDoubleInput));
                m_inputConstraintCone.push_back(std::make_unique<NonnegativeOrthantCone>(m_context, lenDoubleInput));

                if (i >= m_tree.numNonleafNodes()) {
                    hostStateWeightLeaf.insert(hostStateWeightLeaf.end(),
                                               jsonStateWeightLeaf.begin() + (event * lenStateMat),
                                               jsonStateWeightLeaf.begin() + (event * lenStateMat + lenStateMat));
                    hostStateConstraintLeaf.insert(hostStateConstraintLeaf.end(),
                                                   jsonStateConstraintLeaf.begin() + (event * lenDoubleState),
                                                   jsonStateConstraintLeaf.begin() +
                                                   (event * lenDoubleState + lenDoubleState));
                    m_stateConstraintLeafCone.push_back(
                            std::make_unique<NonnegativeOrthantCone>(m_context, lenDoubleState));
                }
            }

            for (size_t i=0; i<m_tree.numNonleafNodes(); i++) {
                if (jsonRiskType == "avar") {
                    m_risk.push_back(std::make_unique<AVaR>(m_context,
                                                            i,
                                                            jsonRiskAlpha,
                                                            m_tree.numChildren(),
                                                            m_tree.childFrom(),
                                                            m_tree.conditionalProbabilities()));
                } else {
                    std::cerr << "Risk type " << jsonRiskType
                        << " is not supported. Supported types include: avar" << std::endl;
                    throw std::invalid_argument("Risk type not supported");
                }
            }

            /** Transfer array data to device */
            m_d_stateDynamics.upload(hostStateDynamics);
            m_d_inputDynamics.upload(hostInputDynamics);
            m_d_stateWeight.upload(hostStateWeight);
            m_d_inputWeight.upload(hostInputWeight);
            m_d_stateWeightLeaf.upload(hostStateWeightLeaf);
            m_d_stateConstraint.upload(hostStateConstraint);
            m_d_inputConstraint.upload(hostInputConstraint);
            m_d_stateConstraintLeaf.upload(hostStateConstraintLeaf);
        }

		/**
		 * Destructor
		 */
		~ProblemData() {}

        /**
         * Getters
         */
        Context& context() { return m_context; }
        size_t numStates() { return m_numStates; }
        size_t numInputs() { return m_numInputs; }
        DeviceVector<real_t>& stateDynamics() { return m_d_stateDynamics; }
        DeviceVector<real_t>& inputDynamics() { return m_d_inputDynamics; }
        DeviceVector<real_t>& stateWeight() { return m_d_stateWeight; }
        DeviceVector<real_t>& inputWeight() { return m_d_inputWeight; }
        DeviceVector<real_t>& stateConstraint() { return m_d_stateConstraint; }
        std::vector<std::unique_ptr<ConvexCone>>& stateConstraintCone() { return m_stateConstraintCone; }
        DeviceVector<real_t>& inputConstraint() { return m_d_inputConstraint; }
        std::vector<std::unique_ptr<ConvexCone>>& inputConstraintCone() { return m_inputConstraintCone; }
        DeviceVector<real_t>& stateConstraintLeaf() { return m_d_stateConstraintLeaf; }
        std::vector<std::unique_ptr<ConvexCone>>& stateConstraintLeafCone() { return m_stateConstraintLeafCone; }
        std::vector<std::unique_ptr<CoherentRisk>>& risk() { return m_risk; }

        /**
         * Debugging
         */
		void print(){
			std::cout << "Number of states: " << m_numStates << std::endl;
            std::cout << "Number of inputs: " << m_numInputs << std::endl;

            size_t len = m_numStates * m_numStates * m_tree.numNodes();
            std::vector<real_t> hostData(len);
            m_d_stateDynamics.download(hostData);
            std::cout << "State dynamics (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_numStates * m_numInputs * m_tree.numNodes();
            hostData.resize(len);
            m_d_inputDynamics.download(hostData);
            std::cout << "Input dynamics (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_numStates * m_numStates * m_tree.numNodes();
            hostData.resize(len);
            m_d_stateWeight.download(hostData);
            std::cout << "State weight (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_numInputs * m_numInputs * m_tree.numNodes();
            hostData.resize(len);
            m_d_inputWeight.download(hostData);
            std::cout << "Input weight (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_numStates * m_numStates * m_tree.numNodes();
            hostData.resize(len);
            m_d_stateWeightLeaf.download(hostData);
            std::cout << "Leaf state weight (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_numStates * 2 * m_tree.numNodes();
            hostData.resize(len);
            m_d_stateConstraint.download(hostData);
            std::cout << "State constraint (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_tree.numNodes();
            std::cout << "State constraint cone dimension: ";
            for (size_t i=0; i<len; i++) {
                std::cout << m_stateConstraintCone[i]->dimension() << " ";
            }
            std::cout << std::endl;

            len = m_numInputs * 2 * m_tree.numNodes();
            hostData.resize(len);
            m_d_inputConstraint.download(hostData);
            std::cout << "Input constraint (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_tree.numNodes();
            std::cout << "Input constraint cone dimension: ";
            for (size_t i=0; i<len; i++) {
                std::cout << m_inputConstraintCone[i]->dimension() << " ";
            }
            std::cout << std::endl;

            len = m_numStates * 2 * m_tree.numNodes();
            hostData.resize(len);
            m_d_stateConstraintLeaf.download(hostData);
            std::cout << "Leaf state constraint (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;

            len = m_tree.numNodes();
            std::cout << "Leaf state constraint cone dimension: ";
            for (size_t i=0; i<len; i++) {
                std::cout << m_stateConstraintLeafCone[i]->dimension() << " ";
            }
            std::cout << std::endl;

            len = m_tree.numNonleafNodes();
            std::cout << "Risk cone dimension: ";
            for (size_t i=0; i<len; i++) {
                std::cout << m_risk[i]->cone().dimension() << " ";
            }
            std::cout << std::endl;
		}
};

#endif
