#ifndef __PROBLEM__
#define __PROBLEM__
#include "../include/stdgpu.h"
#include "tree.cuh"
#include "cones.cuh"
#include <fstream>


/**
 * Store problem data
 * - from default file
 * - from user input
 *
 * Note: `d_` indicates a device pointer
 */
class ProblemData {

	private:
        ScenarioTree& m_tree;  ///< Previously created scenario tree of problem
        Context m_context;  ///< Handle for cublas
        size_t m_numStates = 0;  ///< Total number system states
        size_t m_numInputs = 0;  ///< Total number control inputs
        DeviceVector<real_t> m_d_systemDynamics;  ///< Ptr to
        DeviceVector<real_t> m_d_inputDynamics;  ///< Ptr to
        DeviceVector<real_t> m_d_stateWeight;  ///< Ptr to
        DeviceVector<real_t> m_d_inputWeight;  ///< Ptr to
        DeviceVector<real_t> m_d_stateWeightLeaf;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraint;  ///< Ptr to
        std::vector<ConvexCone*> m_stateConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_inputConstraint;  ///< Ptr to
        std::vector<ConvexCone*> m_inputConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraintLeaf;  ///< Ptr to
        std::vector<ConvexCone*> m_stateConstraintLeafCone;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatE;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatF;  ///< Ptr to
        std::vector<ConvexCone*> m_riskConK;  ///< Ptr to
        DeviceVector<real_t> m_d_riskVecB;  ///< Ptr to

	public:
		/**
		 * Constructor from default JSON file stream
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
            m_numStates = doc["stateSize"].GetInt();
            m_numInputs = doc["inputSize"].GetInt();

            /** Sizes */
            size_t lenStateMat = m_numStates * m_numStates;
            size_t lenInputDynMat = m_numStates * m_numInputs;
            size_t lenInputWgtMat = m_numInputs * m_numInputs;
            size_t lenDoubleState = m_numStates * 2;
            size_t lenDoubleInput = m_numInputs * 2;

            /** Allocate memory on host for JSON data */
            std::vector<real_t> jsonSystemDynamics(lenStateMat * m_tree.numEvents());
            std::vector<real_t> jsonInputDynamics(lenInputDynMat * m_tree.numEvents());
            std::vector<real_t> jsonStateWeight(lenStateMat * m_tree.numEvents());
            std::vector<real_t> jsonInputWeight(lenInputWgtMat * m_tree.numEvents());
            std::vector<real_t> jsonStateWeightLeaf(lenStateMat * m_tree.numEvents());
            std::vector<real_t> jsonStateConstraint(lenDoubleState * m_tree.numEvents());
            std::vector<real_t> jsonInputConstraint(lenDoubleInput * m_tree.numEvents());
            std::vector<real_t> jsonStateConstraintLeaf(lenDoubleState * m_tree.numEvents());
            std::vector<real_t> jsonRiskAlphas(m_tree.numEvents());

            /** Allocate memory on device */
            m_d_systemDynamics.allocateOnDevice(lenStateMat * m_tree.numNodes());
            m_d_inputDynamics.allocateOnDevice(lenInputDynMat * m_tree.numNodes());
            m_d_stateWeight.allocateOnDevice(lenStateMat * m_tree.numNodes());
            m_d_inputWeight.allocateOnDevice(lenInputWgtMat * m_tree.numNodes());
            m_d_stateWeightLeaf.allocateOnDevice(lenStateMat * m_tree.numNodes());
            m_d_stateConstraint.allocateOnDevice(lenDoubleState * m_tree.numNodes());
            m_d_inputConstraint.allocateOnDevice(lenDoubleInput * m_tree.numNodes());
            m_d_stateConstraintLeaf.allocateOnDevice(lenDoubleState * m_tree.numNodes());
            m_d_riskMatE.allocateOnDevice((m_tree.numEvents() * 2 + 1) * (m_tree.numEvents()) * m_tree.numNonleafNodes());
            m_d_riskMatF.allocateOnDevice((m_tree.numEvents() * 2 + 1) * (1) * m_tree.numNonleafNodes());
            m_d_riskVecB.allocateOnDevice((m_tree.numEvents() * 2 + 1) + m_tree.numNonleafNodes());

            /** Store array data from JSON in host memory */
            for (rapidjson::SizeType i = 0; i<lenStateMat; i++) {
                jsonSystemDynamics[i] = doc["systemDynamicsMode0"][i].GetDouble();
                jsonSystemDynamics[i+lenStateMat] = doc["systemDynamicsMode1"][i].GetDouble();
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
            for (rapidjson::SizeType i = 0; i<lenDoubleState; i++) {
                jsonStateConstraint[i] = doc["stateConstraintMode0"][i].GetDouble();
                jsonStateConstraint[i+lenDoubleState] = doc["stateConstraintMode1"][i].GetDouble();
                jsonStateConstraintLeaf[i] = doc["stateConstraintLeafMode0"][i].GetDouble();
                jsonStateConstraintLeaf[i+lenDoubleState] = doc["stateConstraintLeafMode1"][i].GetDouble();
            }
            for (rapidjson::SizeType i = 0; i<lenDoubleInput; i++) {
                jsonInputConstraint[i] = doc["inputConstraintMode0"][i].GetDouble();
                jsonInputConstraint[i+lenDoubleInput] = doc["inputConstraintMode1"][i].GetDouble();
            }
            jsonRiskAlphas[0] = doc["riskMode0"][0].GetDouble();
            jsonRiskAlphas[1] = doc["riskMode1"][0].GetDouble();

            /** Create full arrays on host */
            std::vector<size_t> hostEvents(m_tree.events().capacity());
            m_tree.events().download(hostEvents);
            NullCone nullCone(m_context, 0);
            std::vector<real_t> hostSystemDynamics(lenStateMat, 0.);
            std::vector<real_t> hostInputDynamics(lenInputDynMat, 0.);
            std::vector<real_t> hostStateWeight(lenStateMat, 0.);
            std::vector<real_t> hostInputWeight(lenInputWgtMat, 0.);
            std::vector<real_t> hostStateWeightLeaf(lenStateMat * m_tree.numNonleafNodes(), 0.);
            std::vector<real_t> hostStateConstraint(lenDoubleState, 0.);
            m_stateConstraintCone = {&nullCone};
            std::vector<real_t> hostInputConstraint(lenDoubleInput, 0.);
            m_inputConstraintCone = {&nullCone};
            std::vector<real_t> hostStateConstraintLeaf(lenDoubleState * m_tree.numNonleafNodes(), 0.);
            m_stateConstraintLeafCone = {&nullCone};
            std::vector<real_t> hostRiskMatE;
            std::vector<real_t> hostRiskMatF;
            std::vector<real_t> hostRiskVecB;
            for (size_t i=1; i<m_tree.numNodes(); i++) {
                size_t event = hostEvents[i];
                hostSystemDynamics.insert(hostSystemDynamics.end(), 
                    jsonSystemDynamics.begin() + (event * lenStateMat), 
                    jsonSystemDynamics.begin() + (event * lenStateMat + lenStateMat));
                hostInputDynamics.insert(hostInputDynamics.end(), 
                    jsonInputDynamics.begin() + (event * lenInputDynMat), 
                    jsonInputDynamics.begin() + (event * lenInputDynMat + lenInputDynMat));
                NonnegativeOrthantCone cone(m_context, lenDoubleState);
                m_stateConstraintCone.push_back(&cone);
                std::cout << m_stateConstraintCone[i]->dimension() << " ";
            }
            std::cout << std::endl;

            /** Transfer array data to device */
            m_d_systemDynamics.upload(hostSystemDynamics);
            m_d_inputDynamics.upload(hostInputDynamics);
            // m_d_stateWeight.upload();
            // m_d_inputWeight.upload();
            // m_d_stateWeightLeaf.upload();
            // m_d_stateConstraint.upload();
            // m_d_inputConstraint.upload();
            // m_d_stateConstraintLeaf.upload();
            // m_d_riskMatE.upload();
            // m_d_riskMatF.upload();
            // m_d_riskVecB.upload();
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
        DeviceVector<real_t>& systemDynamics() { return m_d_systemDynamics; }
        DeviceVector<real_t>& inputDynamics() { return m_d_inputDynamics; }
        DeviceVector<real_t>& stateWeight() { return m_d_stateWeight; }
        DeviceVector<real_t>& inputWeight() { return m_d_inputWeight; }
        DeviceVector<real_t>& stateConstraint() { return m_d_stateConstraint; }
        std::vector<ConvexCone*>& stateConstraintCone() { return m_stateConstraintCone; }
        DeviceVector<real_t>& inputConstraint() { return m_d_inputConstraint; }
        std::vector<ConvexCone*>& inputConstraintCone() { return m_inputConstraintCone; }
        DeviceVector<real_t>& stateConstraintLeaf() { return m_d_stateConstraintLeaf; }
        std::vector<ConvexCone*>& stateConstraintLeafCone() { return m_stateConstraintLeafCone; }
        DeviceVector<real_t>& riskMatE() { return m_d_riskMatE; }
        DeviceVector<real_t>& riskMatF() { return m_d_riskMatF; }
        std::vector<ConvexCone*>& riskConK() { return m_riskConK; }
        DeviceVector<real_t>& riskVecB() { return m_d_riskVecB; }

        /**
         * Debugging
         */
		void print(){
			std::cout << "Number of states: " << m_numStates << std::endl;
            std::cout << "Number of inputs: " << m_numInputs << std::endl;

            size_t len = m_numStates * m_numStates * m_tree.numNodes();
            std::vector<real_t> hostData(len);
            m_d_systemDynamics.download(hostData);
            std::cout << "System dynamics (from device): ";
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

            len = m_tree.numNodes();
            std::cout << "State constraint cone dimension: ";
            for (size_t i=1; i<len; i++) {
                std::cout << m_stateConstraintCone[i]->dimension() << " ";
            }
            std::cout << std::endl;
		}
};

#endif
