#include <gtest/gtest.h>
#include "../include/stdgpu.h"


TEST(MatMulTest, AB) {
    Context context;
    size_t aSize = 3;
    size_t bSize = 2;
    std::vector<real_t> matA { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<real_t> matB { 1, 2, 3, 4, 5, 6 };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAB(aSize * bSize);
    gpuMatMul(context, aSize, aSize, bSize, d_matA, d_matB, d_matAB);
    std::vector<real_t> hostData(aSize * bSize);
    d_matAB.download(hostData);
    std::vector<real_t> expectedResult {22, 28, 49, 64, 76, 100};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AtB) {
    Context context;
    size_t aSize = 3;
    size_t bSize = 2;
    std::vector<real_t> matA { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<real_t> matB { 1, 2, 3, 4, 5, 6 };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtB(aSize * bSize);
    gpuMatMul(context, aSize, aSize, bSize, d_matA, d_matB, d_matAtB, true);
    std::vector<real_t> hostData(aSize * bSize);
    d_matAtB.download(hostData);
    std::vector<real_t> expectedResult { 48, 60, 57, 72, 66, 84 };
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, ABt) {
    Context context;
    size_t aSize = 3;
    size_t bSize = 2;
    std::vector<real_t> matA { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<real_t> matB { 1, 2, 3, 4, 5, 6 };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matABt(aSize * bSize);
    gpuMatMul(context, aSize, aSize, bSize, d_matA, d_matB, d_matABt, false, true);
    std::vector<real_t> hostData(aSize * bSize);
    d_matABt.download(hostData);
    std::vector<real_t> expectedResult { 14, 32, 32, 77, 50, 122 };
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AtBt) {
    Context context;
    size_t aSize = 3;
    size_t bSize = 2;
    std::vector<real_t> matA { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<real_t> matB { 1, 2, 3, 4, 5, 6 };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtBt(aSize * bSize);
    gpuMatMul(context, aSize, aSize, bSize, d_matA, d_matB, d_matAtBt, true, true);
    std::vector<real_t> hostData(aSize * bSize);
    d_matAtBt.download(hostData);
    std::vector<real_t> expectedResult { 30, 66, 36, 81, 42, 96 };
    ASSERT_EQ(hostData, expectedResult);
}
