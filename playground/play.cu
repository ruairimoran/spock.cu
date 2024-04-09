#include "../include/stdgpu.h"
#include "wrappers.cuh"

template <typename T>
void printVector(std::vector<T> A) {
    for (size_t i = 0; i < A.size(); i++) {
            std::cout << A[i] << "\t";
    }
    std::cout << "\n\n";
}

template <typename T>
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
    std::vector<real_t> Arow = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12};
    std::vector<real_t> Acol;
    row2col(Acol, Arow, rows, cols);

    // Allocate device memory for A
    DeviceVector<real_t> d_A(Acol);

    // cuBLAS and cuSolver initialization
    Context context;

    // Workspace and information variables
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);

    // SVD calculation
    gpuSvdSetup(context, rows, cols, d_workspace);

    DeviceVector<real_t> d_S(rows);
    DeviceVector<real_t> d_U(rows * rows);
    DeviceVector<real_t> d_Vt(cols * cols);
    bool devInfo = true;

    gpuSvdFactor(context, rows, cols, d_workspace, d_A, d_S, d_U, d_Vt, d_info, devInfo);

    if (devInfo) {
        int info;
        d_info.download(&info);
        if (info != 0) {
            std::cerr << "SVD computation failed with error code " << info << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Compute null space matrix
    DeviceVector<real_t> d_nullspace(d_Vt);

    // Print null space matrix
    std::vector<real_t> nullspace;
    d_nullspace.download(nullspace);
    std::cout << "Nullspace matrix:" << "\n";
    printMatrix(nullspace, cols, cols);

    // Project example vector onto the nullspace of A
    std::vector<real_t> vec = {1, 1, 1};  // Example vector
    DeviceVector<real_t> d_vec(vec);

    DeviceVector<real_t> d_vecProjected(cols);

    // Calculate projection
//    projection = self.__null_space_matrix[i] @ \
//                np.linalg.lstsq(self.__null_space_matrix[i], full_stack, rcond=None)[0]
    gpuMatVecMul(context, cols, cols, d_nullspace, d_vec, d_vecProjected, true);

    // Retrieve result
    std::vector<real_t> vecProjected(cols);
    d_vecProjected.download(vecProjected);
    std::cout << "Projection of the vector onto the nullspace of A:" << "\n";
    printVector(vecProjected);

    // Check result
    DeviceVector<real_t> d_Ax(cols);
    gpuMatVecMul(context, rows, cols, d_A, d_vecProjected, d_Ax);
    std::vector<real_t> Ax(cols);
    d_Ax.download(Ax);
    std::cout << "A * projected vector (should be zeros):" << "\n";
    printVector(Ax);

    return 0;
}
