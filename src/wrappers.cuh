#ifndef WRAPPERS_H
#define WRAPPERS_H

#include "../include/stdgpu.h"


/**
 * Transpose matrix on a device
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param numRows rows of A
 * @param numCols cols of A
 * @param A matrix
 */
template<typename T>
void gpuMatT(Context &context, size_t numRows, size_t numCols, DeviceVector<T> &A) {
    T alpha = 1.0;
    T beta = 0.0;
    DeviceVector<T> transpose(numRows * numCols);
    gpuErrChk(cuLib::geam(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N, numRows, numCols,
                          alpha, A.get(), numCols, beta, A.get(), numRows, transpose.get(), numRows));
    transpose.deviceCopyTo(A);
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

    gpuErrChk(cuLib::geam(context.blas(), opA, opB, numRows, numCols,
                          alpha, A.get(), lda, beta, B.get(), ldb, C.get(), ldc));
}


/**
 * Multiply matrix and vector on a device
 *
 * @tparam T type of elements in the matrices
 * @param context cuBLAS handle
 * @param numRows rows of op(A)
 * @param numCols columns of op(A)
 * @param A input matrix
 * @param x input vector
 * @param y output vector
 * @param transA whether to transpose A
 */
template<typename T>
void gpuMatVecMul(Context &context, size_t numRows, size_t numCols,
                  DeviceVector<T> &A,
                  DeviceVector<T> &x,
                  DeviceVector<T> &y,
                  bool transA = false) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    T alpha = 1.0;
    T beta = 0.0;

    size_t lda = numRows;
    size_t incx = 1;
    size_t incy = 1;

    gpuErrChk(cuLib::gemv(context.blas(), opA, numRows, numCols,
                          alpha, A.get(), lda, x.get(), incx, beta, y.get(), incy));
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
void gpuMatMatMul(Context &context, size_t m, size_t k, size_t n,
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

    gpuErrChk(cuLib::gemm(context.blas(), opA, opB, m, n, k,
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
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskyFactor(Context &context, size_t n,
                       DeviceVector<T> &d_workspace,
                       DeviceVector<T> &d_cholesky,
                       bool devInfo = false) {
    DeviceVector<int> d_info(1);
    const cusolverStatus_t status = cusolverDnDpotrf(context.solver(), CUBLAS_FILL_MODE_LOWER, n,
                                                     d_cholesky.get(), n,
                                                     d_workspace.get(),
                                                     d_workspace.capacity(),
                                                     d_info.get());

    if (devInfo) {
        int info = d_info.fetchElementFromDevice(0);
        if (info != 0) std::cerr << "Cholesky factorization failed with status: " << info << "\n";
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
 * @param devInfo whether to check outcome status for errors
 */
template<typename T>
void gpuCholeskySolve(Context &context, size_t numRowsSol, size_t numColsSol,
                      DeviceVector<T> &d_cholesky,
                      DeviceVector<T> &d_solution,
                      DeviceVector<T> &d_affine,
                      bool devInfo = false) {
    DeviceVector<int> d_info(1);
    d_affine.deviceCopyTo(d_solution);
    const cusolverStatus_t status = cusolverDnDpotrs(context.solver(), CUBLAS_FILL_MODE_LOWER,
                                                     numRowsSol, numColsSol,
                                                     d_cholesky.get(), numRowsSol,
                                                     d_solution.get(), numRowsSol,
                                                     d_info.get());
//    gpuErrChk(cudaDeviceSynchronize());

    if (devInfo) {
        int info = d_info.fetchElementFromDevice(0);
        if (info != 0) std::cerr << "Cholesky solver failed with status: " << info << "\n";
    }

    gpuErrChk(status);
}


template<typename T>
void gpuSvdSetup(Context &context, size_t numRows, size_t numCols, DeviceVector<T> &d_workspace) {
    int workspaceSize;
    gpuErrChk(cuLib::gesvdBufferSize<T>(context.solver(), numRows, numCols, &workspaceSize));
    d_workspace.allocateOnDevice(workspaceSize);
}


template<typename T>
void gpuSvdFactor(
        Context &context,
        size_t numRows,
        size_t numCols,
        DeviceVector<T> &d_workspace,
        DeviceVector<T> &d_A,
        DeviceVector<T> &d_S,
        DeviceVector<T> &d_U,
        DeviceVector<T> &d_Vt,
        bool devInfo = false
) {
    DeviceVector<int> d_info(1);
    const cusolverStatus_t status = cuLib::gesvd(context.solver(), 'N', 'A', numRows, numCols,
                                                 d_A.get(), numRows,
                                                 d_S.get(),
                                                 d_U.get(), numRows,
                                                 d_Vt.get(), numCols,
                                                 d_workspace.get(), d_workspace.capacity(), nullptr, d_info.get());

    if (devInfo) {
        int info = d_info.fetchElementFromDevice(0);
        if (info != 0) {
            std::cerr << "SVD factorisation failed with status " << info << "\n";
        }
    }

    gpuErrChk(status);
}


template<typename T>
void gpuNullspace(
        Context &context,
        size_t numRows,
        size_t numCols,
        DeviceVector<T> &d_A,
        DeviceVector<T> &d_nullspace,
        size_t &nullspaceCols,
        bool devInfo = false) {
    size_t n = numRows * numCols;
    DeviceVector<T> d_copyA(n);
    d_A.deviceCopyTo(d_copyA);

    DeviceVector<T> d_workspace;
    gpuSvdSetup(context, numRows, numCols, d_workspace);

    size_t nVt = numCols * numCols;
    DeviceVector<T> d_S(numCols);
    DeviceVector<T> d_U(numRows * numRows);
    DeviceVector<T> d_Vt(nVt);
    gpuSvdFactor(context, numRows, numCols, d_workspace, d_copyA, d_S, d_U, d_Vt, devInfo);

    std::vector<T> S(numCols);
    d_S.download(S);
    size_t i;
    for (i = 0; i < numCols; i++) { if (S[i] < REAL_PRECISION) break; }
    nullspaceCols = numCols - i;
    size_t idx = nVt - (numCols * nullspaceCols);
    gpuMatT(context, numCols, numCols, d_Vt);
    DeviceVector<T> d_N(d_Vt, idx, nVt - 1);
    d_nullspace.allocateOnDevice(nVt - idx);
    d_N.deviceCopyTo(d_nullspace);
}


/**
 * Solve least squares for x in Ax=b
 *
 * For debugging:
 *      info:
 *          If info=0, the parameters passed to the function are valid
 *          If info<0, the parameter in position -info is invalid
 *      infoArray:
 *          If non-null, every element devInfoArray[i] contain a value V with the following meaning:
 *              V = 0 : the i-th problem was successfully solved
 *              V > 0 : the V-th diagonal element of the Aarray[i] is zero. Aarray[i] does not have full rank.
 */
template<typename T>
void gpuLeastSquares(
        Context &context,
        size_t numRows,
        size_t numCols,
        DeviceVector<T *> &d_ptrsA,
        DeviceVector<T *> &d_ptrsC,
        bool devInfo = false) {
    size_t batchSize = d_ptrsA.capacity();
    int info = 0;
    DeviceVector<int> d_infoArray(batchSize);
    const cublasStatus_t status = cuLib::gels(context.blas(),
                                              CUBLAS_OP_N,
                                              numRows,
                                              numCols,
                                              1,
                                              d_ptrsA.get(),
                                              numRows,
                                              d_ptrsC.get(),
                                              numRows,
                                              &info,
                                              d_infoArray.get(),
                                              batchSize);

    if (devInfo) {
        if (info != 0) {
            std::cerr << "Least squares solve failed with info " << info << "\n";
        }
        std::vector<int> infoArray(batchSize);
        d_infoArray.download(infoArray);
        for (size_t i = 0; i < batchSize; i++) {
            if (infoArray[i] < 0) {
                std::cerr << "Least squares solve failed with infoArray[" << i << "] = " << infoArray[i] << "\n";
            }
        }
    }

    gpuErrChk(status);
}


#endif
