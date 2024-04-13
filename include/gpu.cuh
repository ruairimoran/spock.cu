#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <source_location>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "DeviceVector.cuh"

#ifndef GPU__
#define GPU__


/* ------------------------------------------------------------------------------------
 *  Definitions
 * ------------------------------------------------------------------------------------ */

#define real_t double
#define REAL_PRECISION 1e-12
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost


/* ------------------------------------------------------------------------------------
 *  Error check for Cuda functions
 * ------------------------------------------------------------------------------------ */

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


/* ------------------------------------------------------------------------------------
*  Convert between row- and column-major ordering of vector-stored matrices
* ------------------------------------------------------------------------------------ */

template<typename T>
inline void row2col(std::vector<T> &dstCol, std::vector<T> &srcRow, size_t numRows, size_t numCols) {
    if (numRows * numCols != srcRow.size()) std::cerr << "row2col dimension mismatch" << "\n";
    dstCol.resize(srcRow.size());
    std::vector<T> copySrc(srcRow);
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            dstCol[c * numRows + r] = copySrc[r * numCols + c];
        }
    }
}

template<typename T>
inline void col2row(std::vector<T> &dstRow, std::vector<T> &srcCol, size_t numRows, size_t numCols) {
    if (numRows * numCols != srcCol.size()) std::cerr << "col2row dimension mismatch" << "\n";
    dstRow.resize(srcCol.size());
    std::vector<T> copySrc(srcCol);
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            dstRow[r * numCols + c] = copySrc[c * numRows + r];
        }
    }
}


/* ------------------------------------------------------------------------------------
 *  Namespace that generalises float and double Cuda functions
 * ------------------------------------------------------------------------------------ */

namespace cuLib {
    /**
    * Generic function for cublas `nrm2`
    */
    template<typename T>
    cublasStatus_t nrm2(cublasHandle_t handle, int n, const T *x, int incx, T *result);

    inline cublasStatus_t
    nrm2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
        return cublasSnrm2(handle, n, x, incx, result);
    }

    inline cublasStatus_t
    nrm2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
        return cublasDnrm2(handle, n, x, incx, result);
    }

    /**
    * Generic function for cublas `geam`
    */
    template<typename T>
    cublasStatus_t geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                        T &alpha, T *A, int ldA, T &beta, T *B, int ldB, T *C, int ldC);

    inline cublasStatus_t
    geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
         float &alpha, float *A, int ldA, float &beta, float *B, int ldB, float *C, int ldC) {
        return cublasSgeam(handle, transa, transb, m, n, &alpha, A, ldA, &beta, B, ldB, C, ldC);
    }

    inline cublasStatus_t
    geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
         double &alpha, double *A, int ldA, double &beta, double *B, int ldB, double *C, int ldC) {
        return cublasDgeam(handle, transa, transb, m, n, &alpha, A, ldA, &beta, B, ldB, C, ldC);
    }

    /**
    * Generic function for cublas `gemv`
    */
    template<typename T>
    cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                        T &alpha, T *A, int ldA, T *x, int incx, T &beta, T *y, int incy);

    inline cublasStatus_t
    gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
         float &alpha, float *A, int ldA, float *x, int incx, float &beta, float *y, int incy) {
        return cublasSgemv(handle, trans, m, n, &alpha, A, ldA, x, incx, &beta, y, incy);
    }

    inline cublasStatus_t
    gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
         double &alpha, double *A, int ldA, double *x, int incx, double &beta, double *y, int incy) {
        return cublasDgemv(handle, trans, m, n, &alpha, A, ldA, x, incx, &beta, y, incy);
    }

    /**
    * Generic function for cublas `gemm`
    */
    template<typename T>
    cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                        T &alpha, T *A, int ldA, T *B, int ldB, T &beta, T *C, int ldC);

    inline cublasStatus_t
    gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
         float &alpha, float *A, int ldA, float *B, int ldB, float &beta, float *C, int ldC) {
        return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC);
    }

    inline cublasStatus_t
    gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
         double &alpha, double *A, int ldA, double *B, int ldB, double &beta, double *C, int ldC) {
        return cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC);
    }

    /**
    * Generic function for cusolverDn `potrf_bufferSize`
    */
    template<typename T>
    cusolverStatus_t
    potrfBufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *A, int ldA, int *lwork);

    inline cusolverStatus_t
    potrfBufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int ldA, int *lwork) {
        return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, ldA, lwork);
    }

    inline cusolverStatus_t
    potrfBufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int ldA, int *lwork) {
        return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, ldA, lwork);
    }

    /**
    * Generic function for cusolverDn `potrf`
    */
    template<typename T>
    cusolverStatus_t
    potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *A, int ldA, T *workspace, int lwork,
          int *devInfo);

    inline cusolverStatus_t
    potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int ldA, float *workspace, int lwork,
          int *devInfo) {
        return cusolverDnSpotrf(handle, uplo, n, A, ldA, workspace, lwork, devInfo);
    }

    inline cusolverStatus_t
    potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int ldA, double *workspace, int lwork,
          int *devInfo) {
        return cusolverDnDpotrf(handle, uplo, n, A, ldA, workspace, lwork, devInfo);
    }

    /**
    * Generic function for cusolverDn `potrs`
    */
    template<typename T>
    cusolverStatus_t
    potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, T *A, int ldA, T *B, int ldB,
          int *devInfo);

    inline cusolverStatus_t
    potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float *A, int ldA, float *B, int ldB,
          int *devInfo) {
        return cusolverDnSpotrs(handle, uplo, n, nrhs, A, ldA, B, ldB, devInfo);
    }

    inline cusolverStatus_t
    potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double *A, int ldA, double *B, int ldB,
          int *devInfo) {
        return cusolverDnDpotrs(handle, uplo, n, nrhs, A, ldA, B, ldB, devInfo);
    }

    /**
    * Generic function for cusolverDn `gesvd_bufferSize`
    */
    template<typename T>
    cusolverStatus_t gesvdBufferSize(cusolverDnHandle_t handle, int m, int n, int *lwork);

    template<>
    inline cusolverStatus_t gesvdBufferSize<float>(cusolverDnHandle_t handle, int m, int n, int *lwork) {
        return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
    }

    template<>
    inline cusolverStatus_t gesvdBufferSize<double>(cusolverDnHandle_t handle, int m, int n, int *lwork) {
        return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
    }

    /**
    * Generic function for cusolverDn `gesvd`
    */
    template<typename T>
    cusolverStatus_t gesvd(cusolverDnHandle_t handle, signed char jobU, signed char jobVt, int m, int n,
                           T *A, int ldA,
                           T *S,
                           T *U, int ldU,
                           T *Vt, int ldVt,
                           T *work, int lwork, T *rwork, int *info);

    inline cusolverStatus_t gesvd(cusolverDnHandle_t handle, signed char jobU, signed char jobVt, int m, int n,
                                  float *A, int ldA,
                                  float *S,
                                  float *U, int ldU,
                                  float *Vt, int ldVt,
                                  float *work, int lwork, float *rwork, int *info) {
        return cusolverDnSgesvd(handle, jobU, jobVt, m, n, A, ldA, S, U, ldU, Vt, ldVt, work, lwork, rwork, info);
    }

    inline cusolverStatus_t gesvd(cusolverDnHandle_t handle, signed char jobU, signed char jobVt, int m, int n,
                                  double *A, int ldA,
                                  double *S,
                                  double *U, int ldU,
                                  double *Vt, int ldVt,
                                  double *work, int lwork, double *rwork, int *info) {
        return cusolverDnDgesvd(handle, jobU, jobVt, m, n, A, ldA, S, U, ldU, Vt, ldVt, work, lwork, rwork, info);
    }

    /**
    * Generic function for cublas `gels`
    */
    template<typename T>
    cublasStatus_t
    gels(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, T *Aarray[], int lda,
         T *Carray[], int ldc, int *info, int *devInfoArray, int batchSize);

    inline cublasStatus_t
    gels(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *Aarray[], int lda,
         float *Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
        return cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
    }

    inline cublasStatus_t
    gels(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *Aarray[], int lda,
         double *Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
        return cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
    }
}


#endif
