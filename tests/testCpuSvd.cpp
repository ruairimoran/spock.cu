#include <gtest/gtest.h>
#include "cpu_svd.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
class NoEigenAllocation {
    bool m_previous;
public:
    NoEigenAllocation() : m_previous(spock::detail::setCpuSvdAllocationAllowedForTesting(false)) {}
    ~NoEigenAllocation() { spock::detail::setCpuSvdAllocationAllowedForTesting(m_previous); }
};

template<typename T>
class CpuSvdTest : public testing::Test {};
using Precisions = testing::Types<float,double>;
TYPED_TEST_SUITE(CpuSvdTest,Precisions);

TYPED_TEST(CpuSvdTest, repeatedFactorisationsReuseAllEigenStorage) {
    using T = TypeParam;
    for (int n : {1,3,8}) {
        CpuSvd<T> svd(n);
        std::vector<T> input(n*n);
        const double tolerance = sizeof(T) == sizeof(float) ? 2e-5 : 2e-12;
        for (int repeat = 0; repeat < 12; ++repeat) {
            for (int col = 0; col < n; ++col) for (int row = 0; row < n; ++row)
                input[row+col*n] = T(std::sin(.13*(row+1)*(col+1)+repeat)+(row==col ? 3. : 0.));
            if (repeat == 11) std::fill(input.begin(),input.end(),T(0));
            {
                NoEigenAllocation guard;
                EXPECT_TRUE(svd.factorise(input.data()));
            }
            const auto u=svd.left(),v=svd.right(),s=svd.singularValues();
            for (int col = 0; col < n; ++col) for (int row = 0; row < n; ++row) {
                double reconstructed=0.,innerU=0.,innerV=0.;
                for (int k = 0; k < n; ++k) {
                    reconstructed += double(u[row+k*n])*s[k]*v[col+k*n];
                    innerU += double(u[k+row*n])*u[k+col*n];
                    innerV += double(v[k+row*n])*v[k+col*n];
                }
                EXPECT_NEAR(input[row+col*n],reconstructed,tolerance);
                EXPECT_NEAR(row==col ? 1. : 0.,innerU,tolerance);
                EXPECT_NEAR(row==col ? 1. : 0.,innerV,tolerance);
            }
        }
    }
}

TEST(CpuSvdTest, allocationGuardIsActiveInReleaseBuilds) {
    NoEigenAllocation guard;
    EXPECT_THROW(CpuSvd<double> shouldAllocate(3),std::logic_error);
}

TEST(CpuSvdTest, rejectsInvalidDimensions) {
    EXPECT_THROW(CpuSvd<double> zero(0),std::invalid_argument);
    EXPECT_THROW(CpuSvd<float> negative(-1),std::invalid_argument);
}
} // namespace
