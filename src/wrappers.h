#ifndef WRAPPERS_H
#define WRAPPERS_H

#include "../include/stdgpu.h"

namespace generic {
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
}


/**
 * Transpose matrix on a device
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param numRows rows of A
 * @param numCols cols of A
 * @param A input matrix
 * @param C output matrix
 */
template<typename T>
void gpuMatT(Context &context, size_t numRows, size_t numCols,
             DeviceVector<T> &A,
             DeviceVector<T> &C) {
    T alpha = 1.0;
    T beta = 0.0;
    gpuErrChk(generic::geam(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N, numRows, numCols,
                            alpha, A.get(), numCols, beta, A.get(), numRows, C.get(), numRows));
}


/**
 * Add two matrices on a device
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param numRows outer dimension of op(A)
 * @param numCols outer dimension of op(B)
 * @param A input matrix
 * @param B input matrix
 * @param C output matrix
 * @param transA whether to transpose A
 * @param transB whether to transpose B
 */
template<typename T>
void gpuMatAdd(Context &context, size_t numRows, size_t numCols,
               DeviceVector<T> &A,
               DeviceVector<T> &B,
               DeviceVector<T> &C,
               bool transA = false, bool transB = false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    T alpha = 1.0;
    T beta = 1.0;

    size_t lda = transA ? numCols : numRows;
    size_t ldb = transB ? numCols : numRows;
    size_t ldc = numRows;

    gpuErrChk(generic::geam(context.blas(), opA, opB, numRows, numCols,
                            alpha, A.get(), lda, beta, B.get(), ldb, C.get(), ldc));
}

/**
 * Multiply two matrices on a device
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
template<typename T>
void gpuMatMul(Context &context, size_t m, size_t k, size_t n,
               DeviceVector<T> &A,
               DeviceVector<T> &B,
               DeviceVector<T> &C,
               bool transA = false, bool transB = false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    T alpha = 1.0;
    T beta = 0.0;

    size_t lda = transA ? k : m;
    size_t ldb = transB ? n : k;
    size_t ldc = m;

    gpuErrChk(generic::gemm(context.blas(), opA, opB, m, n, k,
                            alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc));
}


/**
 * Setup Cholesky decomposition workspace
 *
 * @tparam T type of elements in the workspace (i.e., element type of matrix to be decomposed)
 * @param context cusolver handle
 * @param n n=rows=cols of matrix to be decomposed
 * @param d_workspace workspace for Cholesky decomposition
 */
template<typename T>
void gpuCholeskySetup(Context &context, size_t n,
                      DeviceVector<T> &d_workspace) {
    int workspaceSize = 0;
    double *nullPtr = NULL;
    gpuErrChk(cusolverDnDpotrf_bufferSize(context.solver(), CUBLAS_FILL_MODE_LOWER, n,
                                          nullPtr, n, &workspaceSize));
    d_workspace.allocateOnDevice(workspaceSize);
}


/**
 * Cholesky decomposition
 *
 * This function operates a Cholesky decomposition on the given matrix,
 * and overwrites the given matrix with the resulting lower-triangular matrix.
 *
 * @tparam T type of elements in matrix to be decomposed
 * @param context cusolver handle
 * @param n n=rows=cols of matrix
 * @param d_workspace workspace for Cholesky decomposition
 * @param d_cholesky matrix to be operated upon, and overwritten with lower-triangular result
 * @param d_info device storage for outcome status of decomposition
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskyFactor(Context &context, size_t n,
                       DeviceVector<T> &d_workspace,
                       DeviceVector<T> &d_cholesky,
                       DeviceVector<int> &d_info,
                       bool devInfo = false) {
    const cusolverStatus_t status = cusolverDnDpotrf(context.solver(), CUBLAS_FILL_MODE_LOWER, n,
                                                     d_cholesky.get(), n,
                                                     d_workspace.get(),
                                                     d_workspace.capacity(),
                                                     d_info.get());

    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) std::cerr << "Cholesky factorization failed with status: " << info[0] << "\n";
    }

    gpuErrChk(status);
}


/**
 * Solve linear system using previously computed Cholesky decomposition
 *
 * Solve the linear system Ax=b for x, using the previously computed
 * lower-triangular part of the Cholesky decomposition of matrix A.
 *
 * @param context cusolver handle
 * @param numRowsSol number of rows of solution
 * @param numColsSol number of columns of solution
 * @param d_cholesky lower-triangular matrix result of Cholesky decomposition on A
 * @param d_solution solution of linear system, x
 * @param d_affine affine part of linear system, b
 * @param d_info device storage for outcome status of solving system
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskySolve(Context &context, size_t numRowsSol, size_t numColsSol,
                      DeviceVector<T> &d_cholesky,
                      DeviceVector<T> &d_solution,
                      DeviceVector<T> &d_affine,
                      DeviceVector<int> &d_info,
                      bool devInfo = false) {
    d_affine.deviceCopyTo(d_solution);
    const cusolverStatus_t status = cusolverDnDpotrs(context.solver(), CUBLAS_FILL_MODE_LOWER,
                                                     numRowsSol, numColsSol,
                                                     d_cholesky.get(), numRowsSol,
                                                     d_solution.get(), numRowsSol,
                                                     d_info.get());
    gpuErrChk(cudaDeviceSynchronize());

    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) std::cerr << "Cholesky solver failed with status: " << info[0] << "\n";
    }

    gpuErrChk(status);
}


template<typename T>
void gpuSvdSetup(Context &context, int numRows, int numCols, DeviceVector<T> &d_workspace) {
    int workspaceSize;
    gpuErrChk(generic::gesvdBufferSize<T>(context.solver(), numRows, numCols, &workspaceSize));
    d_workspace.allocateOnDevice(workspaceSize);
}


template<typename T>
void gpuSvdFactor(
        Context &context,
        int numRows,
        int numCols,
        DeviceVector<T> &d_workspace,
        DeviceVector<T> &d_A,
        DeviceVector<T> &d_S,
        DeviceVector<T> &d_U,
        DeviceVector<T> &d_Vt,
        DeviceVector<int> &d_info,
        bool devInfo = false
) {
    const cusolverStatus_t status = generic::gesvd(context.solver(), 'A', 'A', numRows, numCols,
                                                   d_A.get(), numRows,
                                                   d_S.get(),
                                                   d_U.get(), numRows,
                                                   d_Vt.get(), numCols,
                                                   d_workspace.get(), d_workspace.capacity(), nullptr, d_info.get()
    );

    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) {
            std::cerr << "SVD factorisation failed with status " << info[0] << "\n";
        }
    }

    gpuErrChk(status);
}


template<typename T>
void gpuNullspace(
        Context &context,
        int numRows,
        int numCols,
        DeviceVector<T> &d_workspace,
        DeviceVector<T> &d_A,
        DeviceVector<T> &d_S,
        DeviceVector<T> &d_U,
        DeviceVector<T> &d_Vt,
        DeviceVector<int> &d_info,
        bool devInfo = false
) {
    gpuSvdSetup(context, numRows, numCols, d_workspace);
    gpuSvdFactor(context, numRows, numCols, d_workspace, d_A, d_S, d_U, d_Vt, d_info, devInfo);

}

#endif
