#include "../include/stdgpu.h"
#include "tree.cuh"
#include "cones.cuh"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdexcept>


/**
 * Repeat original vector n times while concatenating
 * @param[in] originalVec vector to be repeated
 * @param[in] times number of times to repeat vector
 * @return a vector of n originalVec's joined together
*/
template<typename T>
std::vector<T> repeat_n(const std::vector<T>& input, size_t n) {
    std::vector<T> result(input.size() * n);
    auto iter = result.begin();
    for (size_t rep=0; rep<n; rep++, iter+=input.size()) {
        std::copy(input.begin(), input.end(), iter);
    }
    return result;
}


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
        size_t m_numStates = 0;  ///< Total number system states
        size_t m_numInputs = 0;  ///< Total number control inputs
        DeviceVector<real_t> m_d_systemDynamics;  ///< Ptr to
        DeviceVector<real_t> m_d_inputDynamics;  ///< Ptr to
        DeviceVector<real_t> m_d_stateWeight;  ///< Ptr to
        DeviceVector<real_t> m_d_inputWeight;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraint;  ///< Ptr to
        // DeviceVector<ConvexCone*> m_d_stateConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_inputConstraint;  ///< Ptr to
        // DeviceVector<ConvexCone*> m_d_inputConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatE;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatF;  ///< Ptr to
        // DeviceVector<ConvexCone*> m_d_riskConeK;  ///< Ptr to
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
              std::cerr << "Error parsing JSON: " << GetParseError_En(doc.GetParseError()) << std::endl;
              throw std::invalid_argument("Cannot parse JSON file");
            }

            /** Store single element data from JSON in host memory */
            m_numStates = doc["numStates"].GetInt();
            m_numInputs = doc["numInputs"].GetInt();

            /** Allocate memory on host for JSON data */
            std::vector<real_t> hostSystemDynamics(m_numStates * m_numStates * m_tree.numEvents());
            std::vector<real_t> hostInputDynamics(m_numStates * m_numInputs * m_tree.numEvents());
            std::vector<real_t> hostStateWeight(m_numStates * m_numStates * m_tree.numEvents());
            std::vector<real_t> hostInputWeight(m_numInputs * m_numInputs * m_tree.numEvents());
            std::vector<real_t> hostStateWeightTerminal(m_numStates * m_numStates * m_tree.numEvents());
            std::vector<real_t> hostStateConstraintMinMax(m_numStates * 2 * m_tree.numEvents());
            std::vector<real_t> hostInputConstraintMinMax(m_numInputs * 2 * m_tree.numEvents());
            std::vector<real_t> hostStateConstraintTerminalMinMax(m_numStates * 2 * m_tree.numEvents());
            std::vector<real_t> hostRiskAlphas(m_tree.numEvents());

            /** Allocate memory on device */
            m_d_systemDynamics.allocateOnDevice(m_numStates * m_numStates * m_tree.numNodes());
            m_d_inputDynamics.allocateOnDevice(m_numInputs * m_numInputs * m_tree.numNodes());
            m_d_stateWeight.allocateOnDevice(m_numStates * m_numStates * (m_tree.numNodes() + m_tree.numLeafNodes()));
            m_d_inputWeight.allocateOnDevice(m_numInputs * m_numInputs * m_tree.numNodes());
            m_d_stateConstraint.allocateOnDevice(m_numStates * 2 * (m_tree.numNodes() + m_tree.numLeafNodes()));
            // m_d_stateConstraintCone.allocateOnDevice(m_tree.numNodes());
            m_d_inputConstraint.allocateOnDevice(m_numInputs * 2 * m_tree.numNodes());
            // m_d_inputConstraintCone.allocateOnDevice(m_tree.numNodes());
            m_d_riskMatE.allocateOnDevice((m_tree.numEvents() * 2 + 1) * (m_tree.numEvents()) * m_tree.numNodes());
            m_d_riskMatF.allocateOnDevice((m_tree.numEvents() * 2 + 1) * (1) * m_tree.numNodes());
            // m_d_riskConeK.allocateOnDevice(m_tree.numNodes());
            m_d_riskVecB.allocateOnDevice(m_tree.numEvents() * 2 + 1);

            /** Store array data from JSON in host memory */
            size_t len = m_numStates*m_numStates;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                hostSystemDynamics[i] = doc["systemDynamicsMode0"][i].GetDouble();
                hostSystemDynamics[i+len] = doc["systemDynamicsMode1"][i].GetDouble();
                hostStateWeight[i] = doc["stateWeightMode0"][i].GetDouble();
                hostStateWeight[i+len] = doc["stateWeightMode1"][i].GetDouble();
                hostStateWeightTerminal[i] = doc["stateWeightMode0"][i].GetDouble();
                hostStateWeightTerminal[i+len] = doc["stateWeightMode1"][i].GetDouble();
            }
            len = m_numInputs*m_numInputs;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                hostInputDynamics[i] = doc["controlDynamicsMode0"][i].GetDouble();
                hostInputDynamics[i+len] = doc["controlDynamicsMode1"][i].GetDouble();
                hostInputWeight[i] = doc["inputWeightMode0"][i].GetDouble();
                hostInputWeight[i+len] = doc["inputWeightMode1"][i].GetDouble();
            }
            len = m_numStates * 2;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                hostStateConstraintMinMax[i] = doc["stateConstraintMode0"][i].GetDouble();
                hostStateConstraintMinMax[i+len] = doc["stateConstraintMode1"][i].GetDouble();
                hostStateConstraintTerminalMinMax[i] = doc["stateConstraintTerminalMode0"][i].GetDouble();
                hostStateConstraintTerminalMinMax[i+len] = doc["stateConstraintTerminalMode1"][i].GetDouble();
            }
            len = m_numInputs * 2;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                hostInputConstraintMinMax[i] = doc["inputConstraintMode0"][i].GetDouble();
                hostInputConstraintMinMax[i+len] = doc["inputConstraintMode1"][i].GetDouble();
            }
            hostRiskAlphas[0] = doc["riskMode0"][0].GetDouble();
            hostRiskAlphas[0] = doc["riskMode1"][0].GetDouble();

            /** Transfer JSON array data to device */
            m_d_systemDynamics.upload(repeat_n(hostSystemDynamics, m_tree.numNodes()));
            // m_d_inputDynamics.upload();
            // m_d_stateWeight.upload();
            // m_d_inputWeight.upload();
            // m_d_stateConstraint.upload();
            // m_d_stateConstraintCone.upload();
            // m_d_inputConstraint.upload();
            // m_d_inputConstraintCone.upload();
            // m_d_riskMatE.upload();
            // m_d_riskMatF.upload();
            // m_d_riskConeK.upload();
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
        // DeviceVector<ConvexCone*>& stateConstraintCone() { return m_d_stateConstraintCone; }
        DeviceVector<real_t>& inputConstraint() { return m_d_inputConstraint; }
        // DeviceVector<ConvexCone*>& inputConstraintCone() { return m_d_inputConstraintCone; }
        DeviceVector<real_t>& riskMatE() { return m_d_riskMatE; }
        DeviceVector<real_t>& riskMatF() { return m_d_riskMatF; }
        // DeviceVector<ConvexCone*>& riskConeK() { return m_d_riskConeK; }
        DeviceVector<real_t>& riskVecB() { return m_d_riskVecB; }

        /**
         * Debugging
         */
		// void print(){
        //     std::vector<int> hostDataIntNumNodes(m_numNodes);

		// 	std::cout << "Number of nonleaf nodes: " << m_numNonleafNodes << std::endl;

        //     m_d_stages.download(hostDataIntNumNodes);
        //     std::cout << "Stages (from device): ";
        //     for (size_t i=0; i<m_numNodes; i++) {
        //         std::cout << hostDataIntNumNodes[i] << " ";
        //     }
        //     std::cout << std::endl;
		// }
};
