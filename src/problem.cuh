#ifndef __PROBLEM__
#define __PROBLEM__
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
        DeviceVector<real_t> m_d_stateWeightLeaf;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraint;  ///< Ptr to
        DeviceVector<ConvexCone*> m_d_stateConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_inputConstraint;  ///< Ptr to
        DeviceVector<ConvexCone*> m_d_inputConstraintCone;  ///< Ptr to
        DeviceVector<real_t> m_d_stateConstraintLeaf;  ///< Ptr to
        DeviceVector<ConvexCone*> m_d_stateConstraintLeafCone;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatE;  ///< Ptr to
        DeviceVector<real_t> m_d_riskMatF;  ///< Ptr to
        DeviceVector<ConvexCone*> m_d_riskConK;  ///< Ptr to
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
            std::vector<real_t> jsonSystemDynamics(m_numStates * m_numStates * m_tree.numEvents());
            std::vector<real_t> jsonInputDynamics(m_numStates * m_numInputs * m_tree.numEvents());
            std::vector<real_t> jsonStateWeight(m_numStates * m_numStates * m_tree.numEvents());
            std::vector<real_t> jsonInputWeight(m_numInputs * m_numInputs * m_tree.numEvents());
            std::vector<real_t> jsonStateWeightLeaf(m_numStates * m_numStates * m_tree.numEvents());
            std::vector<real_t> jsonStateConstraint(m_numStates * 2 * m_tree.numEvents());
            std::vector<real_t> jsonInputConstraint(m_numInputs * 2 * m_tree.numEvents());
            std::vector<real_t> jsonStateConstraintLeaf(m_numStates * 2 * m_tree.numEvents());
            std::vector<real_t> jsonRiskAlphas(m_tree.numEvents());

            /** Allocate memory on device */
            m_d_systemDynamics.allocateOnDevice(m_numStates * m_numStates * m_tree.numNodes());
            m_d_inputDynamics.allocateOnDevice(m_numInputs * m_numInputs * m_tree.numNodes());
            m_d_stateWeight.allocateOnDevice(m_numStates * m_numStates * m_tree.numNodes());
            m_d_inputWeight.allocateOnDevice(m_numInputs * m_numInputs * m_tree.numNodes());
            m_d_stateWeightLeaf.allocateOnDevice(m_numStates * m_numStates * m_tree.numNodes());
            m_d_stateConstraint.allocateOnDevice(m_numStates * 2 * (m_tree.numNodes() + m_tree.numLeafNodes()));
            m_d_stateConstraintCone.allocateOnDevice(m_tree.numNodes());
            m_d_inputConstraint.allocateOnDevice(m_numInputs * 2 * m_tree.numNodes());
            m_d_stateConstraintLeafCone.allocateOnDevice(m_tree.numNodes());
            m_d_riskMatE.allocateOnDevice((m_tree.numEvents() * 2 + 1) * (m_tree.numEvents()) * m_tree.numNonleafNodes());
            m_d_riskMatF.allocateOnDevice((m_tree.numEvents() * 2 + 1) * (1) * m_tree.numNonleafNodes());
            m_d_riskConK.allocateOnDevice(m_tree.numNonleafNodes());
            m_d_riskVecB.allocateOnDevice((m_tree.numEvents() * 2 + 1) + m_tree.numNonleafNodes());

            /** Store array data from JSON in host memory */
            size_t len = m_numStates*m_numStates;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                jsonSystemDynamics[i] = doc["systemDynamicsMode0"][i].GetDouble();
                jsonSystemDynamics[i+len] = doc["systemDynamicsMode1"][i].GetDouble();
                jsonStateWeight[i] = doc["stateWeightMode0"][i].GetDouble();
                jsonStateWeight[i+len] = doc["stateWeightMode1"][i].GetDouble();
                jsonStateWeightLeaf[i] = doc["stateWeightLeafMode0"][i].GetDouble();
                jsonStateWeightLeaf[i+len] = doc["stateWeightLeafMode1"][i].GetDouble();
            }
            len = m_numInputs*m_numInputs;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                jsonInputDynamics[i] = doc["controlDynamicsMode0"][i].GetDouble();
                jsonInputDynamics[i+len] = doc["controlDynamicsMode1"][i].GetDouble();
                jsonInputWeight[i] = doc["inputWeightMode0"][i].GetDouble();
                jsonInputWeight[i+len] = doc["inputWeightMode1"][i].GetDouble();
            }
            len = m_numStates * 2;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                jsonStateConstraint[i] = doc["stateConstraintMode0"][i].GetDouble();
                jsonStateConstraint[i+len] = doc["stateConstraintMode1"][i].GetDouble();
                jsonStateConstraintLeaf[i] = doc["stateConstraintLeafMode0"][i].GetDouble();
                jsonStateConstraintLeaf[i+len] = doc["stateConstraintLeafMode1"][i].GetDouble();
            }
            len = m_numInputs * 2;
            for (rapidjson::SizeType i = 0; i<len; i++) {
                jsonInputConstraint[i] = doc["inputConstraintMode0"][i].GetDouble();
                jsonInputConstraint[i+len] = doc["inputConstraintMode1"][i].GetDouble();
            }
            jsonRiskAlphas[0] = doc["riskMode0"][0].GetDouble();
            jsonRiskAlphas[0] = doc["riskMode1"][0].GetDouble();

            /** Create full arrays on host */
            // std::vector<real_t> hostSystemDynamics(m_d_systemDynamics.capacity());
            // for (size_t i=0; i<m_tree.numNodes(); i++) {
            //     size_t event = m_tree.events()[i];
            //     hostSystemDynamics.insert(hostSystemDynamics.end(), 
            //         jsonSystemDynamics[event * len], 
            //         jsonSystemDynamics[event * len + len]);
            // }

            /** Transfer array data to device */
            // m_d_systemDynamics.upload(hostSystemDynamics);
            // m_d_inputDynamics.upload();
            // m_d_stateWeight.upload();
            // m_d_inputWeight.upload();
            // m_d_stateWeightLeaf.upload();
            // m_d_stateConstraint.upload();
            // m_d_stateConstraintCone.upload();
            // m_d_inputConstraint.upload();
            // m_d_inputConstraintCone.upload();
            // m_d_stateConstraintLeaf.upload();
            // m_d_riskMatE.upload();
            // m_d_riskMatF.upload();
            // m_d_riskConK.upload();
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
        DeviceVector<ConvexCone*>& stateConstraintCone() { return m_d_stateConstraintCone; }
        DeviceVector<real_t>& inputConstraint() { return m_d_inputConstraint; }
        DeviceVector<ConvexCone*>& inputConstraintCone() { return m_d_inputConstraintCone; }
        DeviceVector<real_t>& stateConstraintLeaf() { return m_d_stateConstraintLeaf; }
        DeviceVector<ConvexCone*>& stateConstraintLeafCone() { return m_d_stateConstraintLeafCone; }
        DeviceVector<real_t>& riskMatE() { return m_d_riskMatE; }
        DeviceVector<real_t>& riskMatF() { return m_d_riskMatF; }
        DeviceVector<ConvexCone*>& riskConK() { return m_d_riskConK; }
        DeviceVector<real_t>& riskVecB() { return m_d_riskVecB; }

        /**
         * Debugging
         */
		void print(){
            size_t len = m_numStates * m_numStates * m_tree.numNodes();
            std::vector<real_t> hostData(len);

			std::cout << "Number of states: " << m_numStates << std::endl;
            std::cout << "Number of inputs: " << m_numInputs << std::endl;

            m_d_systemDynamics.download(hostData);
            std::cout << "System dynamics (from device): ";
            for (size_t i=0; i<len; i++) {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;
		}
};

#endif
