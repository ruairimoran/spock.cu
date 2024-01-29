/* -------------------------------------------------------------
   Compile with
   alias nvcc="/usr/local/cuda-12.3/bin/nvcc"
   nvcc -Wpedantic -I/home/bigboy/Documents/Development/rapidjson/include \
     -std=c++20 main.cu -o a.out; ./a.out
   
   or simply:
   
   cmake .
   make run
  ------------------------------------------------------------- */

#include <stdio.h>
#include <vector>
#include <fstream> 
#include <iostream> 
#include <stdexcept>
#include "../include/stdgpu.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



class ScenarioTree {
	
	private:
		int* m_d_ancestors = 0;  /**< (Device pointer) ancestor indices */
		int* m_d_stages = 0;     /**< (Device pointer) stages */
		int* m_d_childFrom = 0;
		int* m_d_childTo = 0;
		size_t m_numNodes = 0;  /**< number of nodes, incl. root node */
		size_t m_numNonleafNodes = 0; /**< number of nonleaf nodes */
		
		
		/** Allocates memory for tree on GPU */
		void allocateDeviceMemory() 
		{
			size_t nodesBytes = m_numNodes * sizeof(int);
			size_t nonleafNodesBytes = m_numNonleafNodes * sizeof(int);
			gpuErrchk( cudaMalloc((void**)&m_d_ancestors, nodesBytes) );
			gpuErrchk( cudaMalloc((void**)&m_d_stages, nodesBytes) );
			gpuErrchk( cudaMalloc((void**)&m_d_childFrom, nonleafNodesBytes) );
			gpuErrchk( cudaMalloc((void**)&m_d_childTo, nonleafNodesBytes) );
		}
		
		/** Transfer data to device */
		void transferIntDataToDevice(const rapidjson::Value& jsonArray, int* devPtr)
		{
		  size_t arrayLen = jsonArray.Size();
		  std::vector<int> hostData(arrayLen);
		  size_t numBytes = arrayLen * sizeof(int);
		  for (rapidjson::SizeType i = 0; i < arrayLen; i++) {
		  	hostData[i] = jsonArray[i].GetInt();
		  }
		  gpuErrchk( cudaMemcpy(devPtr, hostData.data(), numBytes, H2D) );
		}
		
		
	public:
		
		
		
		/**
		 * Constructor from file stream
		 */
		ScenarioTree(std::ifstream& file)
		{
	  
		  std::string json((std::istreambuf_iterator<char>(file)), 
		                    std::istreambuf_iterator<char>()); 
		  rapidjson::Document doc;
		  doc.Parse(json.c_str()); 
		  
		  if (doc.HasParseError()) { 
		      std::cerr << "Error parsing JSON: " << GetParseError_En(doc.GetParseError()) << std::endl; 
		      throw std::invalid_argument("Cannot parse JSON file");
		  } 
		  
		  const rapidjson::Value& ancestorsJson = doc["ancestors"];
		  const rapidjson::Value& childFromJson = doc["childrenFrom"];
		  m_numNodes = ancestorsJson.Size();
		  m_numNonleafNodes = childFromJson.Size();
		  
		  allocateDeviceMemory();
		  
		  /* Transfer data to device */
		  transferIntDataToDevice(ancestorsJson, m_d_ancestors); 
		  transferIntDataToDevice(doc["stages"], m_d_stages);
		  transferIntDataToDevice(childFromJson, m_d_childFrom);
		  transferIntDataToDevice(doc["childrenTo"], m_d_childTo);
    }
    
    
		/**
		 * Destructor
		 */
		~ScenarioTree(){
			if (m_d_ancestors != 0){
				gpuErrchk( cudaFree(m_d_ancestors) );
				m_d_ancestors = 0;
			}
			if (m_d_stages != 0){
				gpuErrchk( cudaFree(m_d_stages) );
				m_d_stages = 0;
			}
			if (m_d_childFrom != 0) {
				gpuErrchk( cudaFree(m_d_childFrom) );
				m_d_childFrom = 0;
			}
			if (m_d_childTo != 0) {
				gpuErrchk( cudaFree(m_d_childTo) );
				m_d_childTo = 0;
			}			
		}
		
		int numNodes() {
			return m_numNodes;
		}
		
		int* ancestors() {
			return m_d_ancestors;
		}
		
		int* stages() {
			return m_d_stages;
		}
		
	
		void print(){
			// FOR DEBUGGING ONLY!
			std::cout << "Number of ancestors: " << m_numNodes << std::endl; 
			int *hostNodeData = new int[m_numNodes];
			cudaMemcpy(hostNodeData, m_d_ancestors, m_numNodes*sizeof(int), D2H);
			std::cout << "Ancestors (from device): ";
			for (size_t i=0; i<m_numNodes; i++){
				std::cout << hostNodeData[i] << " ";
			}
			std::cout << std::endl;
			
			cudaMemcpy(hostNodeData, m_d_stages, m_numNodes*sizeof(int), D2H);
			std::cout << "Stages (from device): ";
			for (size_t i=0; i<m_numNodes; i++){
				std::cout << hostNodeData[i] << " ";
			}
			std::cout << std::endl;
			
			cudaMemcpy(hostNodeData, m_d_childFrom, m_numNonleafNodes*sizeof(int), D2H);
			std::cout << "Children::from (from device): ";
			for (size_t i=0; i<m_numNonleafNodes; i++){
				std::cout << hostNodeData[i] << " ";
			}
			std::cout << std::endl;
			
			cudaMemcpy(hostNodeData, m_d_childTo, m_numNonleafNodes*sizeof(int), D2H);
			std::cout << "Children::to (from device): ";
			for (size_t i=0; i<m_numNonleafNodes; i++){
				std::cout << hostNodeData[i] << " ";
			}
			std::cout << std::endl;
		} 
		
};
