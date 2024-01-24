#include <gtest/gtest.h>
#include "test_mat_mul.cuh"


TEST(Suite, Test) {
    EXPECT_EQ(test_function(), 0);
}