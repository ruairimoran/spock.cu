#ifndef STANDARD_GPU_INCLUDE
#define STANDARD_GPU_INCLUDE

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define real_t double

dim3 tPerB(512, 512, 512);  ///< standard threads per block
dim3 bPerG(512, 512, 512);  ///< standard blocks per grid

#endif
