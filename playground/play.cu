#include "../include/stdgpu.h"
#include "wrappers.cuh"

template<typename T>
void printVector(std::vector<T> A) {
    for (size_t i = 0; i < A.size(); i++) {
        std::cout << A[i] << "\t";
    }
    std::cout << "\n\n";
}

template<typename T>
void printMatrix(std::vector<T> A, size_t numRows, size_t numCols) {
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            std::cout << A[r * numCols + c] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Matrix A (rows x cols) with rows > cols
    size_t rows = 4;
    size_t cols = 3;
    std::vector<real_t> A = {1, 2, 3,
                             1, 2, 3,
                             1, 2, 3,
                             1, 2, 3};
    row2col(A, A, rows, cols);
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_N;  // nullspace

    Context context;
    size_t NCols = 0;
    gpuNullspace(context, rows, cols, d_A, d_N, NCols, true);
    std::cout << "num nullspace cols: " << NCols << "\n";

    // Check result
    std::vector<real_t> A_(rows * cols);
    d_A.download(A_);
    col2row(A_, A_, rows, cols);
    std::cout << "A:" << "\n";
    printMatrix(A_, rows, cols);

    std::vector<real_t> N_(cols* NCols);
    d_N.download(N_);
    col2row(N_, N_, cols, NCols);
    std::cout << "N:" << "\n";
    printMatrix(N_, cols, NCols);

    size_t nAN = rows * NCols;
    DeviceVector<real_t> d_AN(nAN);
    gpuMatMatMul(context, rows, cols, NCols, d_A, d_N, d_AN);
    std::vector<real_t> AN(nAN);
    d_AN.download(AN);
    std::cout << "A * nullspace (should be zeros):" << "\n";
    printMatrix(AN, rows, NCols);

//    // Workspace and information variables
//    DeviceVector<real_t> d_workspace;
//    DeviceVector<int> d_info(1);
//
//    // SVD calculation
//    gpuSvdSetup(context, rows, cols, d_workspace);
//
//    DeviceVector<real_t> d_U(rows * cols);
//    DeviceVector<real_t> d_S(cols);
//    DeviceVector<real_t> d_Vt(cols * cols);
//    bool devInfo = true;
//
//    gpuSvdFactor(context, rows, cols, d_workspace, d_A, d_S, d_U, d_Vt, d_info, devInfo);
//    std::vector<real_t> S(cols);
//    d_S.download(S);
//    std::cout << "S vector:" << "\n";
//    printVector(S);
//
//    if (devInfo) {
//        int info;
//        d_info.download(&info);
//        if (info != 0) {
//            std::cerr << "SVD computation failed with error code " << info << std::endl;
//            return EXIT_FAILURE;
//        }
//    }
//
//    // Compute null space matrix
//    gpuMatT(context, cols, cols, d_Vt);
//    std::vector<real_t> Vt(cols * cols);
//    d_Vt.download(Vt);
//    std::cout << "Vt matrix:" << "\n";
//    printMatrix(Vt, cols, cols);
//    DeviceVector<real_t> d_nullspace(d_Vt, 6, 8);
//
//    // Print null space matrix
//    std::vector<real_t> nullspace;
//    d_nullspace.download(nullspace);
//    std::cout << "Nullspace matrix:" << "\n";
//    printVector(nullspace);
//
//    // Project example vector onto the nullspace of A
//    std::vector<real_t> vec = {1, 1, 5};  // Example vector
//    DeviceVector<real_t> d_vec(vec);
//
//    // Calculate projection
//    DeviceVector<real_t> d_N(cols);
//    d_nullspace.deviceCopyTo(d_N);
//
//    DeviceVector<real_t> d_vecProjected(cols);
//    std::vector<real_t*> ptrsA = {d_nullspace.get()};
//    DeviceVector<real_t*> d_ptrsA(ptrsA);
//    std::vector<real_t*> ptrsb = {d_vec.get()};
//    DeviceVector<real_t*> d_ptrsb(ptrsb);
//    gpuLeastSquares(context, cols, 1, d_ptrsA, d_ptrsb, true);  // Nz = b
//    DeviceVector<real_t> d_z(d_vec, 0, 0);
//    gpuMatVecMul(context, cols, 1, d_nullspace, d_z, d_vecProjected);  // Nz

    return 0;
}
