#ifndef WRAPPERS_H
#define WRAPPERS_H

#include "../include/stdgpu.h"

#include <source_location>

namespace internal {

template <typename T>
cusolverStatus_t gesvdBufferSizeGeneric(cusolverDnHandle_t handle, int m, int n, int* lwork);

template <>
inline cusolverStatus_t gesvdBufferSizeGeneric<float>(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
}

template <>
inline cusolverStatus_t gesvdBufferSizeGeneric<double>(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
}

inline cusolverStatus_t gesvdGeneric(
    cusolverDnHandle_t handle,
    signed char jobu, signed char jobvt,
    int m, int n,
    float* A, int lda,
    float* S, 
    float* U, int ldu,
    float* VT, int ldvt,
    float* work,
    int lwork,
    float* rwork,
    int* info
) {
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
}

inline cusolverStatus_t gesvdGeneric(
    cusolverDnHandle_t handle,
    signed char jobu, signed char jobvt,
    int m, int n,
    double* A, int lda,
    double* S, 
    double* U, int ldu,
    double* VT, int ldvt,
    double* work,
    int lwork,
    double* rwork,
    int* info
) {
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
}

}

inline void gpuCheckError(cusolverStatus_t status, std::source_location loc = std::source_location::current()) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << status << " (" << loc.file_name() << ":" << loc.line() << ")\n";
        exit(EXIT_FAILURE);
    }
}

/**
 * Handle template type error
 */
template<typename T>
void typeError(T functionDescription) {
    std::cerr << "GPU wrapper error. Maybe incorrect type T? -> " << functionDescription << std::endl;
}


/**
 * Transpose matrix on a device
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param rows rows of A
 * @param cols cols of A
 * @param A input matrix
 * @param C output matrix
 */
template<typename T>
void gpuMatT(Context &context, size_t rows, size_t cols,
             DeviceVector<T> &A,
             DeviceVector<T> &C) {
    T alpha = 1.0;
    T beta = 0.0;
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDgeam(context.blas(),
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              rows, cols,
                              &alpha,
                              A.get(), cols,
                              &beta,
                              A.get(), rows,
                              C.get(), rows));
    } else { typeError(__PRETTY_FUNCTION__); }
}


/**
 * Add two matrices on a device
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
               DeviceVector<T> &A,
               DeviceVector<T> &B,
               DeviceVector<T> &C,
               bool transA = false, bool transB = false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    T alpha = 1.0;
    T beta = 1.0;

    size_t lda = transA ? cols : rows;
    size_t ldb = transB ? cols : rows;
    size_t ldc = rows;

    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDgeam(context.blas(),
                              opA,
                              opB,
                              rows, cols,
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

    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDgemm(context.blas(),
                              opA,
                              opB,
                              m, n, k,
                              &alpha,
                              A.get(), lda,
                              B.get(), ldb,
                              &beta,
                              C.get(), ldc));
    } else { typeError(__PRETTY_FUNCTION__); }
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
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cusolverDnDpotrf_bufferSize(context.solver(),
                                              CUBLAS_FILL_MODE_LOWER,
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
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cusolverDnDpotrf(context.solver(),
                                   CUBLAS_FILL_MODE_LOWER,
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
 * Solve the linear system Ax=b for x, using the previously computed
 * lower-triangular part of the Cholesky decomposition of matrix A.
 *
 * @param context cusolver handle
 * @param solRows number of rows of solution
 * @param solCols number of columns of solution
 * @param d_cholesky lower-triangular matrix result of Cholesky decomposition on A
 * @param d_solution solution of linear system, x
 * @param d_affine affine part of linear system, b
 * @param d_info device storage for outcome status of solving system
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskySolve(Context &context, size_t solRows, size_t solCols,
                      DeviceVector<T> &d_cholesky,
                      DeviceVector<T> &d_solution,
                      DeviceVector<T> &d_affine,
                      DeviceVector<int> &d_info,
                      bool devInfo = false) {
    d_affine.deviceCopyTo(d_solution);
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cusolverDnDpotrs(context.solver(),
                                   CUBLAS_FILL_MODE_LOWER,
                                   solRows,
                                   solCols,
                                   d_cholesky.get(),
                                   solRows,
                                   d_solution.get(),
                                   solRows,
                                   d_info.get()));
    } else { typeError(__PRETTY_FUNCTION__); }
    gpuErrChk(cudaDeviceSynchronize());
    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) std::cerr << "Cholesky solver failed with status: " << info[0] << std::endl;
    }
}

template <typename T>
void gpuSVDSetup(Context& context, int numRows, int numCols, DeviceVector<T>& d_workspace) {
    int workspaceSize;
    gpuCheckError(internal::gesvdBufferSizeGeneric<T>(
        context.solver(),
        numRows, numCols,
        &workspaceSize
    ));
    d_workspace.allocateOnDevice(workspaceSize);
}

template <typename T>
void gpuSVDFactorise(
    Context& context,
    int numRows,
    int numCols,
    DeviceVector<T>& d_workspace,
    DeviceVector<T>& d_A,
    DeviceVector<T>& d_S,
    DeviceVector<T>& d_U,
    DeviceVector<T>& d_VT,
    DeviceVector<int>& d_info,
    bool devInfo = false
) {
    const cusolverStatus_t status = internal::gesvdGeneric(
        context.solver(),
        'A', 'A',
        numRows, numCols,
        d_A.get(), numRows,
        d_S.get(),
        d_U.get(), numRows,
        d_VT.get(), numCols,
        d_workspace.get(), d_workspace.capacity(),
        nullptr,
        d_info.get()
    );

    if (devInfo) {
        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] != 0) {
            std::cerr << "SVD factorisation failed with status " << info[0] << "\n";
        }
    }

    gpuCheckError(status);
}

#endif
