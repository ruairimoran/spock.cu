#ifndef __STANDARD_GPU_INCLUDE__
#define __STANDARD_GPU_INCLUDE__


#define real_t double
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"
#include "CublasContext.cuh"

/**
 * Check for errors when calling CUDA or cuBLAS functions
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

template<typename T>
inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << std::endl;
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublas error. Name: " << cublasGetStatusName(code)
                      << ", string: " << cublasGetStatusString(code)
                      << ", file: " << file << ", line: " << line << std::endl;
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "cusolver error. Status: " << code
                      << ", file: " << file << ", line: " << line << std::endl;
            if (abort) exit(code);
        }
    }
}


#endif
