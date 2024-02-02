#include <gtest/gtest.h>
#include "test_mat_mul.cuh"


TEST(MatMulTest, Test1) {
    EXPECT_EQ(test_function(), 0);
}