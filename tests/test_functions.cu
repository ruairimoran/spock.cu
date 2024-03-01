#include <gtest/gtest.h>
#include "../include/stdgpu.h"


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
