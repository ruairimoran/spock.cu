#include <gtest/gtest.h>
#include "../include/gpu.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t, T *, T *, T *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectPolyhedron(size_t, T *, T *);


class KernelsTest : public testing::Test {
protected:
    KernelsTest() = default;
};

TEMPLATE_WITH_TYPE_T
void projectRectangle(T epsilon) {
    size_t n = 1025;
    T ub = 4;
    T lb = -ub;
    T oob = 0.1;
    std::vector<T> v(n, ub + oob);
    for (size_t i = 0; i < n; i+=2) v[i] *= -1;  // include some negatives
    for (size_t i = 0; i < n; i+=3) v[i] *= .5;  // include some within bounds
    DTensor<T> d_v(v, n);
    DTensor<T> d_lb(std::vector<T>(n, lb), n);  // lower bound
    DTensor<T> d_ub(std::vector<T>(n, ub), n);  // upper bound
    k_projectRectangle<<<numBlocks(n, TPB), TPB>>>(n, d_v.raw(), d_lb.raw(), d_ub.raw());
    EXPECT_NEAR(d_v(0, 0, 0), -(ub+oob)*.5, epsilon);
    for (size_t i = 0; i < n; i++) {
        EXPECT_TRUE(d_v(i, 0, 0) >= lb);
        EXPECT_TRUE(d_v(i, 0, 0) <= ub);
    }
}

TEST_F(KernelsTest, projectRectangle) {
    projectRectangle<float>(TEST_PRECISION_LOW);
    projectRectangle<double>(TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void projectPolyhedron(T epsilon) {
    size_t n = 1025;
    T ub = 4;
    T oob = 0.1;
    std::vector<T> v(n, ub + oob);
    for (size_t i = 0; i < n; i+=2) v[i] *= -1;  // include some negatives
    for (size_t i = 0; i < n; i+=3) v[i] *= .5;  // include some within bound
    DTensor<T> d_v(v, n);
    DTensor<T> d_ub(std::vector<T>(n, ub), n);  // upper bound
    k_projectPolyhedron<<<numBlocks(n, TPB), TPB>>>(n, d_v.raw(), d_ub.raw());
    EXPECT_NEAR(d_v(0, 0, 0), -(ub+oob)*.5, epsilon);
    for (size_t i = 0; i < n; i++) {
        EXPECT_TRUE(d_v(i, 0, 0) <= ub);
    }
}

TEST_F(KernelsTest, projectPolyhedron) {
    projectPolyhedron<float>(TEST_PRECISION_LOW);
    projectPolyhedron<double>(TEST_PRECISION_HIGH);
}
