#ifndef STANDARD_GPU_INCLUDE
#define STANDARD_GPU_INCLUDE

#define real_t double
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

#include <vector>
#include <cublas_v2.h>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"
#include "CublasContext.cuh"

#endif
