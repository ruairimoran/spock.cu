#ifndef STANDARD_GPU_INCLUDE
#define STANDARD_GPU_INCLUDE


#define real_t double
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

#include <vector>
#include <cublas_v2.h>
#include <iostream>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"
#include "CublasContext.cuh"

/**
 * Check for errors when calling CUDA or cuBLAS functions
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
template <typename T>
inline void gpuAssert(T code, const char *file, int line, bool abort=true) {
    if constexpr(std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "GPU assert error. String: " << cudaGetErrorString(code)
                << ", file: " << file << ", line: " << line << std::endl;
            if (abort) exit(code);
        }
    } else if constexpr(std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "GPU assert error. Name: " << cublasGetStatusName(code)
                << ", string: " << cublasGetStatusString(code)
                << ", file: " << file << ", line: " << line << std::endl;
            if (abort) exit(code);
        }
    }
}

/**
 * Add two matrices on a device
 *
 * The cuBLAS library uses column-major order for data layout. We use row-major order.
 * They can be easily exchanged: a row-major matrix is the transpose of it's column-major data.
 * If op(A) is false, op(A) = A. If true, op(A) = A.T.
 * In terms of row-major matrices, this function computes C = op(A) + op(B).
 * To do this, we use cuBLAS to compute C.T = op(A.T) + op(B.T).
 * The transpose of A, B, and C do not have to be computed,
 * as cuBLAS reads and writes column-major data, which is the transposes of the given row-major data.
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param rows outer dimension of op(A)
 * @param cols outer dimension of op(B)
 * @param A input matrix
 * @param B input matrix
 * @param C output matrix
 * @param transA whether to transpose A
 * @param transB whether to transpose B
 */
template <typename T>
void gpuMatAdd(Context& context, size_t rows, size_t cols,
               DeviceVector<T>& A, DeviceVector<T>& B, DeviceVector<T>& C, bool transA=false, bool transB=false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const T alpha = 1.0;
    const T beta = 1.0;
    size_t lda, ldb = 0;
    if (not transA && not transB) { lda = cols; ldb = cols; }
    if (transA && not transB) { lda = rows; ldb = cols; }
    if (not transA && transB) { lda = cols; ldb = rows; }
    if (transA && transB) { lda = rows; ldb = rows; }
    size_t ldc = cols;
    if constexpr(std::is_same_v<T, double>) {
        gpuErrChk(cublasDgeam(context.handle(), opA, opB, cols, rows,
                              &alpha, A.get(), lda, &beta, B.get(), ldb, C.get(), ldc));
    } else {
        std::cerr << "gpuMatAdd() error. Maybe incorrect type T? -> " << __PRETTY_FUNCTION__ << std::endl;
    }
}

/**
 * Multiply two matrices on a device
 *
 * The cuBLAS library uses column-major order for data layout. We use row-major order.
 * They can be easily exchanged: a row-major matrix is the transpose of it's column-major data.
 * If op(A) is false, op(A) = A. If true, op(A) = A.T.
 * In terms of row-major matrices, this function computes C = op(A) * op(B).
 * To do this, we use cuBLAS to compute C.T = op(B.T) * op(A.T).
 * The transpose of A, B, and C do not have to be computed,
 * as cuBLAS reads and writes column-major data, which is the transposes of the given row-major data.
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param m outer dimension of op(A)
 * @param k inner dimensions of op(A) and op(B)
 * @param n outer dimension of op(B)
 * @param A input matrix
 * @param B input matrix
 * @param C output matrix
 * @param transA whether to transpose A
 * @param transB whether to transpose B
 */
template <typename T>
void gpuMatMul(Context& context, size_t m, size_t k, size_t n,
               DeviceVector<T>& A, DeviceVector<T>& B, DeviceVector<T>& C, bool transA=false, bool transB=false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const T alpha = 1.0;
    const T beta = 0.0;
    size_t lda, ldb = 0;
    if (not transA && not transB) { lda = k; ldb = n; }
    if (transA && not transB) { lda = m; ldb = n; }
    if (not transA && transB) { lda = k; ldb = k; }
    if (transA && transB) { lda = m; ldb = k; }
    size_t ldc = n;
    if constexpr(std::is_same_v<T, double>) {
        gpuErrChk(cublasDgemm(context.handle(), opB, opA, n, m, k,
                              &alpha, B.get(), ldb, A.get(), lda, &beta, C.get(), ldc));
    } else {
        std::cerr << "gpuMatMul() error. Maybe incorrect type T? -> " << __PRETTY_FUNCTION__ << std::endl;
    }
}


#endif
