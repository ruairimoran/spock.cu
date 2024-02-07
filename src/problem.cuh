#include "../include/stdgpu.h"
#include "tree.cuh"
#include "cones.cuh"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdexcept>


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
        DeviceVector<ConvexCone*> m_d_stateConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_inputConstraint;  ///< Ptr to
        DeviceVector<ConvexCone*> m_d_inputConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatE;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatF;  ///< Ptr to
        DeviceVector<ConvexCone*> m_d_riskConeK;  ///< Ptr to
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
            std::vector<real_t> hostStateConstraint;
            std::vector<real_t> hostInputConstraint;

            /** Allocate memory on device */
            // m_d_stages.allocateOnDevice(m_numNodes);

            /** Store array data from JSON in host memory */
            // for (rapidjson::SizeType i = 0; i<m_numNodes; i++) {
            //     if (i < m_numNonleafNodes) {
            //         hostChildrenFrom[i] = doc["childrenFrom"][i].GetInt();
            //     }
            //     hostProbabilities[i] = doc["probabilities"][i].GetDouble();
            // }

            /** Transfer JSON array data to device */
            // m_d_stages.upload(hostStages);
        }

		/**
		 * Destructor
		 */
		~ProblemData() {}

        /**
         * Getters
         */
        // bool isMarkovian() { return m_isMarkovian; }
        // int numNonleafNodes() { return m_numNonleafNodes; }
        // DeviceVector<int>& stages() { return m_d_stages; }

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
