#include <gtest/gtest.h>
#include "cache.cuh"
#include <cmath>
#include <type_traits>

// Expose scratch storage only in this test; the production cache API stays
// unchanged. normM must leave M*r available until the safeguard dot product.
template<typename T>
class MetricProductTestCache : public Cache<T> {
public:
    using Cache<T>::Cache;
    using Cache<T>::dotM;
    using Cache<T>::normM;
    DTensor<T> &metricProduct() { return *this->m_d_workDot; }
};

template<typename T>
void checkCachedMetricProduct() {
    const std::string path = SPOCK_TEST_DATA_DIR;
    ScenarioTree<T> tree(path);
    ProblemData<T> data(tree);
    MetricProductTestCache<T> cache(tree, data);
    const size_t primal = cache.sizePrimal(), dual = cache.sizeDual();
    const size_t size = primal + dual;
    std::vector<T> linearOperator;
    DTensor<T>::parseFromFile(path + "test_metricOperator" + tree.fpFileExt()).download(linearOperator);
    ASSERT_EQ(linearOperator.size(), primal * dual);
    DTensor<T> residual(size), direction(size);
    const size_t allocated = Session::getInstance().totalAllocatedBytes();
    const long double tolerance = std::is_same_v<T, float> ? 3e-5L : 2e-12L;

    for (int sample = 0; sample < 8; ++sample) {
        std::vector<T> r(size), d(size);
        for (size_t i = 0; i < size; ++i) {
            r[i] = T(std::cos(0.7 * double(i + 1)));
            d[i] = T(std::sin(0.3 * double(i + 2)));
            if (sample == 0 || (sample == 2 && i >= primal) || (sample == 3 && i < primal)) r[i] = T(0);
            if (sample == 4) d[i] = r[i];
            if (sample == 5) d[i] = -r[i];
            if (sample == 6) { r[i] *= T(1e-3); d[i] *= T(1e3); }
            if (sample == 7) { r[i] *= T(1e3); d[i] *= T(1e-3); }
        }
        // CPU reference: M*r = [r_p - alpha L' r_d; r_d - alpha L r_p].
        // Accumulate in extended precision independently of the CUDA adjoint.
        std::vector<long double> expected(r.begin(), r.end());
        for (size_t col = 0; col < primal; ++col) {
            for (size_t row = 0; row < dual; ++row) {
                const long double coefficient = static_cast<long double>(data.stepSize())
                    * linearOperator[col * dual + row];
                expected[col] -= coefficient * r[primal + row];
                expected[primal + row] -= coefficient * r[col];
            }
        }
        long double expectedNormSquared = 0, expectedCross = 0;
        long double productScale = 1;
        for (size_t i = 0; i < size; ++i) {
            expectedNormSquared += r[i] * expected[i];
            expectedCross += d[i] * expected[i];
            productScale = std::max(productScale, std::abs(expected[i]));
        }
        residual.upload(r);
        direction.upload(d);
        const T originalCross = cache.dotM(residual, direction);
        const T norm = cache.normM(residual, residual);
        std::vector<T> before, after;
        cache.metricProduct().download(before);
        const T reusedCross = direction.dotF(cache.metricProduct());
        cache.metricProduct().download(after);
        EXPECT_EQ(before, after) << "The safeguard dot must not modify M*r";
        ASSERT_TRUE(std::isfinite(norm));
        ASSERT_TRUE(std::isfinite(reusedCross));
        for (size_t i = 0; i < size; ++i)
            EXPECT_NEAR(before[i], expected[i], tolerance * productScale) << "sample " << sample << ", entry " << i;
        const long double scale = 1 + std::abs(expectedNormSquared) + std::abs(expectedCross);
        EXPECT_NEAR(originalCross, expectedCross, tolerance * scale) << "sample " << sample;
        EXPECT_NEAR(reusedCross, expectedCross, tolerance * scale) << "sample " << sample;
        const long double rho = static_cast<long double>(norm) * norm - reusedCross;
        EXPECT_NEAR(rho, expectedNormSquared - expectedCross, tolerance * scale) << "sample " << sample;
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
}

TEST(MetricProductTest, cachedSafeguardFloat) { checkCachedMetricProduct<float>(); }
TEST(MetricProductTest, cachedSafeguardDouble) { checkCachedMetricProduct<double>(); }
