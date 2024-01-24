#include <stdio.h>
#include <vector>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__global__ void setvals(int *x, int n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) x[i] = 3 * i;
}




class ScenarioTree {
    public:
        int *dev_ancestors = 0;
        
        /**
         * Constructor of ScenarioTree
         */
        ScenarioTree(std::vector<int>* ancestors) {
            int n_ancestors = ancestors->size();
            // allocate memory for ancestors
            size_t n_ancestors_bytes = n_ancestors * sizeof(int);
            gpuErrchk(
                cudaMalloc((void**)&dev_ancestors, n_ancestors_bytes)  );
            gpuErrchk(
                cudaMemcpy(dev_ancestors, ancestors->data(), n_ancestors_bytes, cudaMemcpyHostToDevice)  );
            
            // do something silly
            setvals<<<n_ancestors, 1>>>(dev_ancestors, n_ancestors);
        }
        
        /**
         * Destructor
         */
        ~ScenarioTree(){
            if (dev_ancestors != 0) {
                cudaFree(dev_ancestors);
            }
            dev_ancestors = 0;
        }
        
};




int main(void)
{

    // We have the ancestors stored on the host
  std::vector<int> h_anc = {0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3};
  
  // We create a scenario tree: we pass the host-based data
  // and the tree stores a pointer to the device-based data
    ScenarioTree tree(&h_anc);
    
    
    // We can now get the device data back just to print them,
    // but we won't have to do this, except for debugging
    int *ancestors_copy_dev_data = new int[h_anc.size()];
    cudaMemcpy(ancestors_copy_dev_data, tree.dev_ancestors, h_anc.size()*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 4; i++){
        printf("z[%d]: %d\n", i, ancestors_copy_dev_data[i]);
  }    
  
  delete []ancestors_copy_dev_data;
    
}