#include <gtest/gtest.h>
#include "anderson.cuh"
#include <array>

namespace {

template<typename T>
class AndersonSvdTest : public testing::Test {};
using Precisions = testing::Types<float, double>;
TYPED_TEST_SUITE(AndersonSvdTest, Precisions);

// This upper-triangular history has non-symmetric right singular vectors.
// Its ridge coefficients have a closed form, so the test detects treating
// GPUtils' V^T as V without relying on a second SVD implementation.
TYPED_TEST(AndersonSvdTest, repeatedFitsMatchAnalyticRidge) {
    using T = TypeParam;
    constexpr size_t rows = 8, memory = 3;
    const auto before = Session::getInstance().totalAllocatedBytes();
    for (size_t streamIndex : std::array<size_t, 2>{0, 1}) {
        if (streamIndex >= Session::getInstance().numStreams()) continue;
        for (bool rolling : {false, true}) {
            SCOPED_TRACE(rolling ? "rolling QR" : "full QR");
            SCOPED_TRACE(streamIndex);
            std::vector<T> c(rows * memory, T(0)), y(rows * memory, T(0));
            std::vector<T> residual{T(3), T(-1), T(.5), T(.25), T(0), T(0), T(0), T(0)};
            for (size_t col = 0; col < memory; ++col) c[col * rows + col] = T(1);
            DTensor<T> dC(c, rows, memory), dY(y, rows, memory), r(residual, rows);
            DTensor<T> qr(rows, memory), rhs(rows), gamma(memory), direction(rows);
            for (auto *tensor : {&dC, &dY, &r, &qr, &rhs, &gamma, &direction})
                tensor->setStreamIdx(streamIndex);
            AndersonOptions options;
            options.regularization = .1;
            options.qr.incremental = rolling;
            RegularisedAnderson<T> fit(dC, dY, qr, rhs, gamma, options);
            const auto allocated = Session::getInstance().totalAllocatedBytes();
            for (int iteration = 0; iteration < 6; ++iteration) {
                const double a = .5 + .5 * iteration;
                std::fill(y.begin(), y.end(), T(0));
                y[0] = T(1); y[rows] = T(a); y[rows + 1] = T(1); y[2 * rows + 2] = T(4);
                dY.upload(y);
                // A new problem may replace all history data. Reset the QR
                // cache while retaining the same SVD object and input storage.
                fit.reset();
                const auto &result = fit.compute(r, direction);
                ASSERT_EQ(result.status, AndersonFitStatus::accepted);
                ASSERT_EQ(result.selectedColumns, (std::vector<size_t>{0, 1, 2}));
                // scale=4, so normalisation makes the raw ridge penalty 16*lambda.
                const double penalty = 16 * options.regularization;
                const double determinant = (1 + penalty) * (a * a + 1 + penalty) - a * a;
                const std::array<double, 3> expected{
                    (3 + 3 * penalty + a) / determinant,
                    (3 * a * penalty - 1 - penalty) / determinant,
                    2 / (16 + penalty)};
                const double tolerance = std::is_same_v<T, float> ? 5e-5 : 2e-12;
                std::vector<T> deviceGamma, actual;
                gamma.download(deviceGamma);
                direction.download(actual);
                for (size_t col = 0; col < memory; ++col) {
                    EXPECT_NEAR(result.coefficients[col], expected[col], tolerance);
                    EXPECT_NEAR(deviceGamma[col], expected[col], tolerance);
                }
                for (size_t row = 0; row < rows; ++row)
                    EXPECT_NEAR(actual[row], -double(residual[row]) - (row < memory ? expected[row] : 0.), tolerance);
                EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
            }
        }
    }
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), before);
}

}  // namespace
