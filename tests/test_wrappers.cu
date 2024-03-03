#include <gtest/gtest.h>
#include "../include/stdgpu.h"
#include "../src/wrappers.h"


TEST(MatAddTest, APlusB) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{1, 3,
                             5, 7,
                             9, 11};
    std::vector<real_t> matB{0, 2,
                             4, 6,
                             8, 10};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAPlusB(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAPlusB);
    std::vector<real_t> hostData(rows * cols);
    d_matAPlusB.download(hostData);
    std::vector<real_t> expectedResult{1, 5,
                                       9, 13,
                                       17, 21};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, AtPlusB) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{1, 5, 9,
                             3, 7, 11};
    std::vector<real_t> matB{0, 2,
                             4, 6,
                             8, 10};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtPlusB(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAtPlusB, true);
    std::vector<real_t> hostData(rows * cols);
    d_matAtPlusB.download(hostData);
    std::vector<real_t> expectedResult{1, 5,
                                       9, 13,
                                       17, 21};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, APlusBt) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{1, 3,
                             5, 7,
                             9, 11};
    std::vector<real_t> matB{0, 4, 8,
                             2, 6, 10};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAPlusBt(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAPlusBt, false, true);
    std::vector<real_t> hostData(rows * cols);
    d_matAPlusBt.download(hostData);
    std::vector<real_t> expectedResult{1, 5,
                                       9, 13,
                                       17, 21};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, AtPlusBt) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{1, 5, 9,
                             3, 7, 11};
    std::vector<real_t> matB{0, 4, 8,
                             2, 6, 10};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtPlusBt(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAtPlusBt, true, true);
    std::vector<real_t> hostData(rows * cols);
    d_matAtPlusBt.download(hostData);
    std::vector<real_t> expectedResult{1, 5,
                                       9, 13,
                                       17, 21};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, APlusAStoredInA) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{1, 3,
                             5, 7,
                             9, 11};
    DeviceVector<real_t> d_matA(matA);
    gpuMatAdd(context, rows, cols, d_matA, d_matA, d_matA);
    std::vector<real_t> hostData(rows * cols);
    d_matA.download(hostData);
    std::vector<real_t> expectedResult{2, 6,
                                       10, 14,
                                       18, 22};
    ASSERT_EQ(hostData, expectedResult);
}


TEST(MatMulTest, AB) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;
    std::vector<real_t> matA{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9,
                             10, 11, 12};
    std::vector<real_t> matB{1, 2,
                             3, 4,
                             5, 6};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAB(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matAB);
    std::vector<real_t> hostData(m * n);
    d_matAB.download(hostData);
    std::vector<real_t> expectedResult{22, 28,
                                       49, 64,
                                       76, 100,
                                       103, 136};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AtB) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;
    std::vector<real_t> matA{1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12};
    std::vector<real_t> matB{1, 2,
                             3, 4,
                             5, 6};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtB(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matAtB, true);
    std::vector<real_t> hostData(m * n);
    d_matAtB.download(hostData);
    std::vector<real_t> expectedResult{61, 76,
                                       70, 88,
                                       79, 100,
                                       88, 112};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, ABt) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;
    std::vector<real_t> matA{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9,
                             10, 11, 12};
    std::vector<real_t> matB{1, 2, 3,
                             4, 5, 6};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matABt(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matABt, false, true);
    std::vector<real_t> hostData(m * n);
    d_matABt.download(hostData);
    std::vector<real_t> expectedResult{14, 32,
                                       32, 77,
                                       50, 122,
                                       68, 167};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AtBt) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;
    std::vector<real_t> matA{1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12};
    std::vector<real_t> matB{1, 2, 3,
                             4, 5, 6};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtBt(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matAtBt, true, true);
    std::vector<real_t> hostData(m * n);
    d_matAtBt.download(hostData);
    std::vector<real_t> expectedResult{38, 83,
                                       44, 98,
                                       50, 113,
                                       56, 128};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AAStoredInA) {
    Context context;
    size_t n = 3;
    std::vector<real_t> matA{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
    DeviceVector<real_t> d_matA(matA);
    gpuMatMul(context, n, n, n, d_matA, d_matA, d_matA);
    std::vector<real_t> hostData(n * n);
    d_matA.download(hostData);
    std::vector<real_t> expectedResult{30, 36, 42,
                                       66, 81, 96,
                                       102, 126, 150};
    ASSERT_EQ(hostData, expectedResult);
}


TEST(CholeskyDecompositionTest, Factor) {
    Context context;
    size_t n = 3;
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);
    std::vector<real_t> A{1, 0, 0,
                          2, 3, 0,
                          4, 5, 6};
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_C(n * n);
    gpuMatMul(context, n, n, n, d_A, d_A, d_C, false, true);
    //* setup */
    gpuCholeskySetup(context, n, d_workspace);
    //* factor */
    gpuCholeskyFactor(context, n, d_workspace, d_C, d_info, true);
    std::vector<real_t> hostData(n * n);
    d_C.download(hostData);
    for (size_t i: {0, 3, 4, 6, 7, 8}) {
        ASSERT_EQ(hostData[i], A[i]);
    }
}

TEST(CholeskyDecompositionTest, FactorAndSolve) {
    Context context;
    size_t n = 3;
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);
    std::vector<real_t> A{1, 0, 0,
                          2, 3, 0,
                          4, 5, 6};
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_C(n * n);
    gpuMatMul(context, n, n, n, d_A, d_A, d_C, false, true);
    //* setup */
    gpuCholeskySetup(context, n, d_workspace);
    //* factor */
    gpuCholeskyFactor(context, n, d_workspace, d_C, d_info, true);
    //* solve */
    DeviceVector<real_t> d_x(n);
    DeviceVector<real_t> d_b(std::vector<real_t>({17, 97, 281}));
    gpuCholeskySolve(context, n, d_C, d_x, d_b, d_info, true);
    std::vector<real_t> hostData(n);
    d_x.download(hostData);
    std::vector<real_t> expectedResult{1, 2, 3};
    ASSERT_EQ(hostData, expectedResult);
}
