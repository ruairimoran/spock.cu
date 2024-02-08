#ifndef STANDARD_GPU_INCLUDE
#define STANDARD_GPU_INCLUDE

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define real_t double

#include <vector>
#include <cublas_v2.h>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"
#include "CublasContext.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif
