#ifndef STANDARD_GPU_INCLUDE
#define STANDARD_GPU_INCLUDE

#include <vector>
#include <cublas_v2.h>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"
#include "CublasContext.cuh"

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define real_t double

#endif
