#include <gtest/gtest.h>
#include "anderson.cuh"
#include <array>

namespace {

template<typename T>
class AndersonQrBackendTest : public testing::Test {};
using Precisions = testing::Types<float, double>;
TYPED_TEST_SUITE(AndersonQrBackendTest, Precisions);

// Orthogonal, moving history columns give an independent closed-form ridge
// solution. They also expose reversal of the physical newest-first ordering.
// Frequent Gram checks and rebuilds exercise reuse of the small device scratch.
TYPED_TEST(AndersonQrBackendTest, rollingAndFullFitsMatchAnalyticRidgeThroughRebuilds) {
    using T = TypeParam;
    constexpr size_t rows = 17, memory = 3;
    const auto before = Session::getInstance().totalAllocatedBytes();
    for (size_t streamIndex : std::array<size_t, 2>{0, 1}) {
        if (streamIndex >= Session::getInstance().numStreams()) continue;
        for (bool rolling : {false, true}) {
            SCOPED_TRACE(rolling ? "rolling QR" : "full QR");
            SCOPED_TRACE(streamIndex);
            DTensor<T> c(rows, memory), y(rows, memory), q(rows, memory);
            DTensor<T> rhs(rows), gamma(memory), residual(rows), direction(rows);
            for (auto *tensor : {&c, &y, &q, &rhs, &gamma, &residual, &direction})
                tensor->setStreamIdx(streamIndex);
            AndersonOptions options;
            options.regularization = .05;
            options.qr.incremental = rolling;
            options.qr.rebuildInterval = 3;
            options.qr.checkInterval = 1;
            RegularisedAnderson<T> fit(c, y, q, rhs, gamma, options);
            const auto allocated = Session::getInstance().totalAllocatedBytes();
            const auto qPointer = q.raw(), rhsPointer = rhs.raw();
            for (size_t step = 0; step < 14; ++step) {
                SCOPED_TRACE(step);
                std::vector<T> cHost(rows * memory, T(0)), yHost(rows * memory, T(0)), rHost(rows);
                std::array<size_t, memory> indices;
                std::array<double, memory> weights;
                for (size_t row = 0; row < rows; ++row) rHost[row] = T(std::cos(.4 * row + .1 * step));
                for (size_t col = 0; col < memory; ++col) {
                    const size_t stamp = step + rows - col;
                    indices[col] = stamp % rows;
                    weights[col] = 1 + .25 * (stamp % 5);
                    yHost[col * rows + indices[col]] = T(weights[col]);
                    cHost[col * rows + indices[col]] = T(.2 * weights[col]);
                }
                // A deficient history must fall back and the next full-rank
                // history must recover using the same GPUtils workspace.
                const bool deficient = step == 7;
                if (deficient) std::fill(yHost.begin(), yHost.end(), T(0));
                c.upload(cHost); y.upload(yHost); residual.upload(rHost);
                if (step == 10) fit.reset();
                fit.historyChanged();
                const auto &result = fit.compute(residual, direction);
                ASSERT_EQ(result.status, deficient ? AndersonFitStatus::rankZero : AndersonFitStatus::accepted);
                const double scale = *std::max_element(weights.begin(), weights.end());
                std::vector<double> expectedDirection(rows);
                for (size_t row = 0; row < rows; ++row) expectedDirection[row] = -double(rHost[row]);
                const double tolerance = std::is_same_v<T, float> ? 2e-5 : 2e-12;
                std::vector<T> actualGamma, actualDirection;
                gamma.download(actualGamma); direction.download(actualDirection);
                for (size_t col = 0; col < memory; ++col) {
                    const double weight = weights[col];
                    const double expected = deficient ? 0 : weight * double(rHost[indices[col]]) /
                        (weight * weight + options.regularization * scale * scale);
                    EXPECT_NEAR(expected, result.coefficients[col], tolerance);
                    EXPECT_NEAR(expected, actualGamma[col], tolerance);
                    expectedDirection[indices[col]] -= double(cHost[col * rows + indices[col]]) * expected;
                }
                for (size_t row = 0; row < rows; ++row)
                    EXPECT_NEAR(expectedDirection[row], actualDirection[row], tolerance);
                if (!deficient) {
                    const auto reuses = fit.qrStatistics().reuses;
                    ASSERT_TRUE(fit.compute(residual, direction).accepted());
                    if (rolling) EXPECT_EQ(reuses + 1, fit.qrStatistics().reuses);
                }
                EXPECT_EQ(qPointer, q.raw()); EXPECT_EQ(rhsPointer, rhs.raw());
                EXPECT_EQ(allocated, Session::getInstance().totalAllocatedBytes());
            }
            EXPECT_GT(fit.qrStatistics().rebuilds, 0);
            if (rolling) {
                EXPECT_GT(fit.qrStatistics().updates, 0);
                EXPECT_GT(fit.qrStatistics().orthogonalityChecks, 0);
            }
        }
    }
    EXPECT_EQ(before, Session::getInstance().totalAllocatedBytes());
}

} // namespace
