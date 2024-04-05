#include <gtest/gtest.h>

#include <array>
#include "../include/stdgpu.h"
#include "../src/wrappers.h"

TEST(MatTransposeTest, At) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{1, 3,
                             5, 7,
                             9, 11};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matAt(rows * cols);
    gpuMatT(context, rows, cols, d_matA, d_matAt);
    std::vector<real_t> hostData(rows * cols);
    d_matAt.download(hostData);
    std::vector<real_t> expectedResult{1, 5, 9,
                                       3, 7, 11};
    ASSERT_EQ(hostData, expectedResult);
}


TEST(MatAddTest, APlusB) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{
            1, 3,
            5, 7,
            9, 11
    };
    std::vector<real_t> matB{
            0, 2,
            4, 6,
            8, 10
    };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAPlusB(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAPlusB);
    std::vector<real_t> hostData(rows * cols);
    d_matAPlusB.download(hostData);
    std::vector<real_t> expectedResult{
            1, 5,
            9, 13,
            17, 21
    };
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, AtPlusB) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{
            1, 3,
            5, 7,
            9, 11
    };
    std::vector<real_t> matB{
            0, 4, 8,
            2, 6, 10
    };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtPlusB(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAtPlusB, true);
    std::vector<real_t> hostData(rows * cols);
    d_matAtPlusB.download(hostData);
    std::vector<real_t> expectedResult{
            1, 9, 17,
            5, 13, 21
    };
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, APlusBt) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{
            1, 5, 9,
            3, 7, 11
    };
    std::vector<real_t> matB{
            0, 2,
            4, 6,
            8, 10
    };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAPlusBt(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAPlusBt, false, true);
    std::vector<real_t> hostData(rows * cols);
    d_matAPlusBt.download(hostData);
    std::vector<real_t> expectedResult{
            1, 9, 17,
            5, 13, 21
    };
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, AtPlusBt) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{
            1, 3,
            5, 7,
            9, 11
    };
    std::vector<real_t> matB{
            0, 2,
            4, 6,
            8, 10
    };
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtPlusBt(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA, d_matB, d_matAtPlusBt, true, true);
    std::vector<real_t> hostData(rows * cols);
    d_matAtPlusBt.download(hostData);
    std::vector<real_t> expectedResult{
            1, 9, 17,
            5, 13, 21
    };
    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatAddTest, APlusAStoredInA) {
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    std::vector<real_t> matA{
            1, 5, 9,
            3, 7, 11
    };
    DeviceVector<real_t> d_matA(matA);
    gpuMatAdd(context, rows, cols, d_matA, d_matA, d_matA);
    std::vector<real_t> hostData(rows * cols);
    d_matA.download(hostData);
    std::vector<real_t> expectedResult{
            2, 10, 18,
            6, 14, 22
    };
    ASSERT_EQ(hostData, expectedResult);
}


TEST(MatMulTest, AB) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;

    std::vector<real_t> matA{
            1, 4, 7, 10,
            2, 5, 8, 11,
            3, 6, 9, 12
    };

    std::vector<real_t> matB{
            1, 3, 5,
            2, 4, 6,
    };

    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAB(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matAB);
    std::vector<real_t> hostData(m * n);
    d_matAB.download(hostData);

    std::vector<real_t> expectedResult{
            22, 49, 76, 103,
            28, 64, 100, 136
    };

    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AtB) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;

    std::vector<real_t> matA{
            1, 5, 9,
            2, 6, 10,
            3, 7, 11,
            4, 8, 12
    };

    std::vector<real_t> matB{
            1, 3, 5,
            2, 4, 6
    };

    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtB(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matAtB, true);
    std::vector<real_t> hostData(m * n);
    d_matAtB.download(hostData);

    std::vector<real_t> expectedResult{
            61, 70, 79, 88,
            76, 88, 100, 112
    };

    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, ABt) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;

    std::vector<real_t> matA{
            1, 4, 7, 10,
            2, 5, 8, 11,
            3, 6, 9, 12
    };

    std::vector<real_t> matB{
            1, 4,
            2, 5,
            3, 6
    };

    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matABt(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matABt, false, true);
    std::vector<real_t> hostData(m * n);
    d_matABt.download(hostData);

    std::vector<real_t> expectedResult{
            14, 32, 50, 68,
            32, 77, 122, 167
    };

    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AtBt) {
    Context context;
    size_t m = 4;
    size_t k = 3;
    size_t n = 2;

    std::vector<real_t> matA{
            1, 5, 9,
            2, 6, 10,
            3, 7, 11,
            4, 8, 12
    };

    std::vector<real_t> matB{
            1, 4,
            2, 5,
            3, 6
    };

    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAtBt(m * n);
    gpuMatMul(context, m, k, n, d_matA, d_matB, d_matAtBt, true, true);
    std::vector<real_t> hostData(m * n);
    d_matAtBt.download(hostData);

    std::vector<real_t> expectedResult{
            38, 44, 50, 56,
            83, 98, 113, 128
    };

    ASSERT_EQ(hostData, expectedResult);
}

TEST(MatMulTest, AAStoredInA) {
    Context context;
    size_t n = 3;

    std::vector<real_t> matA{
            1, 4, 7,
            2, 5, 8,
            3, 6, 9
    };

    DeviceVector<real_t> d_matA(matA);
    gpuMatMul(context, n, n, n, d_matA, d_matA, d_matA);
    std::vector<real_t> hostData(n * n);
    d_matA.download(hostData);

    std::vector<real_t> expectedResult{
            30, 66, 102,
            36, 81, 126,
            42, 96, 150
    };

    ASSERT_EQ(hostData, expectedResult);
}


TEST(CholeskyDecompositionTest, Factor) {
    Context context;
    size_t n = 3;
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);
    std::vector<real_t> A{
            1, 2, 4,
            0, 3, 5,
            0, 0, 6
    };
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_C(n * n);
    gpuMatMul(context, n, n, n, d_A, d_A, d_C, false, true);
    /** setup */
    gpuCholeskySetup(context, n, d_workspace);
    /** factor */
    gpuCholeskyFactor(context, n, d_workspace, d_C, d_info, true);
    std::vector<real_t> hostData(n * n);
    d_C.download(hostData);
    for (size_t i: {0, 1, 2, 4, 5, 8}) {
        EXPECT_EQ(hostData[i], A[i]);
    }
}

TEST(CholeskyDecompositionTest, FactorAndSolveWithVec) {
    Context context;
    size_t n = 3;
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);
    std::vector<real_t> A{1, 2, 4,
                          0, 3, 5,
                          0, 0, 6};
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_C(n * n);
    gpuMatMul(context, n, n, n, d_A, d_A, d_C, false, true);
    /** setup */
    gpuCholeskySetup(context, n, d_workspace);
    /** factor */
    gpuCholeskyFactor(context, n, d_workspace, d_C, d_info, true);
    /** solve */
    DeviceVector<real_t> d_x(n);
    DeviceVector<real_t> d_b(std::vector<real_t>({17, 97, 281}));
    gpuCholeskySolve(context, n, 1, d_C, d_x, d_b, d_info, true);
    std::vector<real_t> hostData(n);
    d_x.download(hostData);
    std::vector<real_t> expectedResult{1, 2, 3};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(CholeskyDecompositionTest, FactorAndSolveWithMat) {
    Context context;
    size_t n = 3;
    size_t len = n * n;
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);
    std::vector<real_t> A{1, 2, 4,
                          0, 3, 5,
                          0, 0, 6};
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_C(len);
    gpuMatMul(context, n, n, n, d_A, d_A, d_C, false, true);
    /** setup */
    gpuCholeskySetup(context, n, d_workspace);
    /** factor */
    gpuCholeskyFactor(context, n, d_workspace, d_C, d_info, true);
    /** solve */
    DeviceVector<real_t> d_X(len);
    DeviceVector<real_t> d_Xt(len);
    DeviceVector<real_t> d_B(std::vector<real_t>({37, 44, 51, 215, 253, 291, 635, 739, 843}));
    DeviceVector<real_t> d_Bt(len);
    gpuMatT(context, n, n, d_B, d_Bt);
    gpuCholeskySolve(context, n, n, d_C, d_Xt, d_Bt, d_info, true);
    gpuMatT(context, n, n, d_Xt, d_X);
    std::vector<real_t> hostData(len);
    d_X.download(hostData);
    std::vector<real_t> expectedResult{1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(hostData, expectedResult);
}

TEST(SvdTest, Factorise) {

    Context context;

    constexpr size_t rows = 5;
    constexpr size_t cols = 4;

    const std::vector<real_t> A{
            1, 0, 0, 0, 2,
            0, 0, 3, 0, 0,
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0
    };
    DeviceVector<real_t> d_A{A};

    DeviceVector<real_t> d_S{rows};
    DeviceVector<real_t> d_U{rows * rows};
    DeviceVector<real_t> d_Vt{cols * cols};
    DeviceVector<int> d_info{1};

    DeviceVector<real_t> d_workspace;
    gpuSvdSetup(context, rows, cols, d_workspace);

    gpuSvdFactor(
            context,
            rows, cols,
            d_workspace,
            d_A,
            d_S,
            d_U,
            d_Vt,
            d_info,
            true
    );

    std::vector<real_t> S;
    d_S.download(S);
    std::vector<real_t> U;
    d_U.download(U);
    std::vector<real_t> Vt;
    d_Vt.download(Vt);

    constexpr std::array<real_t, cols> expectedS = {
            3, 2.23606797749979, 2, 0
    };

    constexpr std::array<real_t, rows * rows> expectedU = {
            0, 0, -1, 0, 0,
            -0.4472135954999579, 0, 0, 0, -0.8944271909999159,
            0, -1, 0, 0, 0,
            0, 0, 0, 1, 0,
            -0.8944271909999159, 0, 0, 0, 0.4472135954999579
    };

    constexpr std::array<real_t, cols * cols> expectedVt = {
            0, -1, 0, 0,
            -1, 0, 0, 0,
            0, 0, 0, -1,
            0, 0, -1, 0
    };

    for (size_t i = 0; i < cols; i++) {
        EXPECT_NEAR(S[i], expectedS[i], REAL_PRECISION);
    }
    for (size_t i = 0; i < rows * rows; i++) {
        EXPECT_NEAR(U[i], expectedU[i], REAL_PRECISION);
    }
    for (size_t i = 0; i < cols * cols; i++) {
        EXPECT_NEAR(Vt[i], expectedVt[i], REAL_PRECISION);
    }
}

TEST(NullspaceTest, Nullspace) {
    Context context;
    size_t rows = 5;
    size_t cols = 4;

    std::vector<real_t> A{
            1, 0, 0, 0, 2,
            0, 0, 3, 0, 0,
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0
    };
    DeviceVector<real_t> d_A{A};

    DeviceVector<real_t> d_S{rows};
    DeviceVector<real_t> d_U{rows * rows};
    DeviceVector<real_t> d_Vt{cols * cols};
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info{1};
    gpuNullspace(context, rows, cols, d_workspace,
                 d_A, d_S, d_U, d_Vt, d_info, true);

    std::vector<real_t> S;
    d_S.download(S);
    std::vector<real_t> U;
    d_U.download(U);
    std::vector<real_t> Vt;
    d_Vt.download(Vt);
}
