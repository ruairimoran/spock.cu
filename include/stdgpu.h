#ifndef __STANDARD_GPU_INCLUDE__
#define __STANDARD_GPU_INCLUDE__


#define real_t double
#define REAL_PRECISION 1e-12
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <source_location>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"
#include "CublasContext.cuh"

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), std::source_location::current()); }

template<typename T>
inline void gpuAssert(T code, std::source_location loc, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << loc.file_name() << ", line: " << loc.line() << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublas error. Name: " << cublasGetStatusName(code)
                      << ", string: " << cublasGetStatusString(code)
                      << ", file: " << loc.file_name() << ", line: " << loc.line() << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cusolverStatus_t>) {
        if (code != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "cusolver error. Status: " << code
                      << ", file: " << loc.file_name() << ", line: " << loc.line() << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}


#endif
