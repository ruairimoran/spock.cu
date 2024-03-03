#ifndef __WRAPPERS__
#define __WRAPPERS__

#include "../include/stdgpu.h"


/**
 * Handle template type error
 */
template<typename T>
void typeError(T functionDescription) {
    std::cerr << "GPU wrapper error. Maybe incorrect type T? -> " << functionDescription << std::endl;
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
template<typename T>
void gpuMatAdd(Context &context, size_t rows, size_t cols,
               DeviceVector<T> &A, DeviceVector<T> &B, DeviceVector<T> &C, bool transA = false, bool transB = false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const T alpha = 1.0;
    const T beta = 1.0;
    size_t lda, ldb = 0;
    if (not transA && not transB) {
        lda = cols;
        ldb = cols;
    }
    if (transA && not transB) {
        lda = rows;
        ldb = cols;
    }
    if (not transA && transB) {
        lda = cols;
        ldb = rows;
    }
    if (transA && transB) {
        lda = rows;
        ldb = rows;
    }
    size_t ldc = cols;
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDgeam(context.blas(),
                              opA,
                              opB,
                              cols, rows,
                              &alpha,
                              A.get(), lda,
                              &beta,
                              B.get(), ldb,
                              C.get(), ldc));
    } else { typeError(__PRETTY_FUNCTION__); }
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
template<typename T>
void gpuMatMul(Context &context, size_t m, size_t k, size_t n,
               DeviceVector<T> &A, DeviceVector<T> &B, DeviceVector<T> &C, bool transA = false, bool transB = false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const T alpha = 1.0;
    const T beta = 0.0;
    size_t lda, ldb = 0;
    if (not transA && not transB) {
        lda = k;
        ldb = n;
    }
    if (transA && not transB) {
        lda = m;
        ldb = n;
    }
    if (not transA && transB) {
        lda = k;
        ldb = k;
    }
    if (transA && transB) {
        lda = m;
        ldb = k;
    }
    size_t ldc = n;
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDgemm(context.blas(),
                              opB,
                              opA,
                              n, m, k,
                              &alpha,
                              B.get(), ldb,
                              A.get(), lda,
                              &beta,
                              C.get(), ldc));
    } else { typeError(__PRETTY_FUNCTION__); }
}


/**
 * Setup Cholesky decomposition workspace
 *
 * @tparam T type of elements in the workspace (i.e., element type of matrix to be decomposed)
 * @param context cusolver handle
 * @param n dimension of solution vector
 * @param d_workspace workspace for Cholesky decomposition
 */
template<typename T>
void gpuCholeskySetup(Context &context, size_t n, DeviceVector<T> &d_workspace) {
    int workspaceSize = 0;
    double *nullPtr = NULL;
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cusolverDnDpotrf_bufferSize(context.solver(),
                                              CUBLAS_FILL_MODE_UPPER,
                                              n,
                                              nullPtr,
                                              n,
                                              &workspaceSize));
    } else { typeError(__PRETTY_FUNCTION__); }
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
 * @param n dimension of solution vector
 * @param d_workspace workspace for Cholesky decomposition
 * @param d_cholesky matrix to be operated upon, and overwritten with lower-triangular result
 * @param d_info device storage for outcome status of decomposition
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskyFactor(Context &context, size_t n, DeviceVector<T> &d_workspace,
                       DeviceVector<T> &d_cholesky,
                       DeviceVector<int> &d_info,
                       bool devInfo = false) {
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cusolverDnDpotrf(context.solver(),
                                   CUBLAS_FILL_MODE_UPPER,
                                   n,
                                   d_cholesky.get(),
                                   n,
                                   d_workspace.get(),
                                   d_workspace.capacity(),
                                   d_info.get()));
    } else { typeError(__PRETTY_FUNCTION__); }
    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) std::cerr << "Cholesky factorization failed with status: " << info[0] << std::endl;
    }
}


/**
 * Solve linear system using previously computed Cholesky decomposition
 *
 * Solve the linear system Ax=b for vector x, using the previously computed
 * lower-triangular part of the Cholesky decomposition of matrix A.
 *
 * @param context cusolver handle
 * @param n dimension of solution vector
 * @param d_cholesky lower-triangular matrix result of Cholesky decomposition on A
 * @param d_solution solution of linear system, x
 * @param d_affine affine vector of linear system, b
 * @param d_info device storage for outcome status of solving system
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskySolve(Context &context, size_t n,
                      DeviceVector<T> &d_cholesky,
                      DeviceVector<T> &d_solution,
                      DeviceVector<T> &d_affine,
                      DeviceVector<int> &d_info,
                      bool devInfo = false) {
    d_affine.deviceCopyTo(d_solution);
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cusolverDnDpotrs(context.solver(),
                                   CUBLAS_FILL_MODE_UPPER,
                                   n,
                                   1,
                                   d_cholesky.get(),
                                   n,
                                   d_solution.get(),
                                   n,
                                   d_info.get()));
    } else { typeError(__PRETTY_FUNCTION__); }
    gpuErrChk(cudaDeviceSynchronize());
    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) std::cerr << "Cholesky solver failed with status: " << info[0] << std::endl;
    }
}


#endif
