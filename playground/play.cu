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
    Context context;

    // Matrix A (rows x cols) with rows >= cols
    size_t rows = 4;
    size_t cols = 3;
    std::vector<real_t> A = {1, 2, 3,
                             1, 2, 3,
                             1, 2, 3,
                             1, 2, 3};
    row2col(A, A, rows, cols);
    DeviceVector<real_t> d_A(context, A);
    DeviceVector<real_t> d_N(context, 0);  // nullspace

    size_t NCols = 0;
    gpuNullspace(context, rows, cols, d_A, d_N, NCols, true);
    std::cout << "num nullspace cols: " << NCols << "\n";

    // Check result
    std::vector<real_t> A_(rows * cols);
    d_A.download(A_);
    col2row(A_, A_, rows, cols);
    std::cout << "A:" << "\n";
    printMatrix(A_, rows, cols);

    std::vector<real_t> N_(cols * NCols);
    d_N.download(N_);
    col2row(N_, N_, cols, NCols);
    std::cout << "N:" << "\n";
    printMatrix(N_, cols, NCols);

    size_t nAN = rows * NCols;
    DeviceVector<real_t> d_AN(context, nAN);
    gpuMatMatMul(context, rows, cols, NCols, d_A, d_N, d_AN);
    std::vector<real_t> AN(nAN);
    d_AN.download(AN);
    std::cout << "A * nullspace (should be zeros):" << "\n";
    printMatrix(AN, rows, NCols);

    return 0;
}
