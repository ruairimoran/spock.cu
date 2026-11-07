#include <gtest/gtest.h>
#include "cache.cuh"
#include <cstdlib>

namespace {

// Rebind both library handles together. Nonblocking streams deliberately
// remove the legacy-default-stream ordering that could hide a missing stream
// argument in a kernel launch or host transfer.
class StreamEnvironment : public testing::Environment {
    std::vector<cudaStream_t> m_originalBlas, m_originalSolver, m_nonblocking;
public:
    void SetUp() override {
        const bool single = std::getenv("SPOCK_TEST_SINGLE_STREAM") != nullptr;
        Session::setStreams(single ? 1 : 2);
        if (!std::getenv("SPOCK_TEST_NONBLOCKING_STREAM")) return;
        auto &session = Session::getInstance();
        m_originalBlas.resize(session.numStreams());
        m_originalSolver.resize(session.numStreams());
        m_nonblocking.resize(session.numStreams());
        for (size_t i = 0; i < session.numStreams(); ++i) {
            gpuErrChk(cublasGetStream(session.cuBlasHandle(i), &m_originalBlas[i]));
            gpuErrChk(cusolverDnGetStream(session.cuSolverHandle(i), &m_originalSolver[i]));
            gpuErrChk(cudaStreamCreateWithFlags(&m_nonblocking[i], cudaStreamNonBlocking));
            gpuErrChk(cublasSetStream(session.cuBlasHandle(i), m_nonblocking[i]));
            gpuErrChk(cusolverDnSetStream(session.cuSolverHandle(i), m_nonblocking[i]));
        }
    }

    void TearDown() override {
        auto &session = Session::getInstance();
        for (size_t i = 0; i < m_nonblocking.size(); ++i) {
            gpuErrChk(cudaStreamSynchronize(m_nonblocking[i]));
            gpuErrChk(cublasSetStream(session.cuBlasHandle(i), m_originalBlas[i]));
            gpuErrChk(cusolverDnSetStream(session.cuSolverHandle(i), m_originalSolver[i]));
            gpuErrChk(cudaStreamDestroy(m_nonblocking[i]));
        }
    }
};

[[maybe_unused]] auto *streamEnvironment = testing::AddGlobalTestEnvironment(new StreamEnvironment);

template<typename T>
class ExecutionTestCache : public Cache<T> {
public:
    using Cache<T>::Cache;
    using Cache<T>::backup;
    using Cache<T>::restore;
    using Cache<T>::computeResidual;
    using Cache<T>::acceptCandidate;

    DTensor<T> &iterate() { return *this->m_d_iterate; }
    DTensor<T> &candidate() { return *this->m_d_iterateCandidate; }
    DTensor<T> &residual() { return *this->m_d_residual; }
    DTensor<T> &historyC() { return *this->m_d_andIterateMatrix; }
    DTensor<T> &historyY() { return *this->m_d_andResidualMatrix; }
    DTensor<T> &deltaIterate() { return *this->m_d_deltaIterate; }
    DTensor<T> &deltaResidual() { return *this->m_d_deltaResidual; }
    void shiftHistory() {
        this->m_validAndersonHistory = 0;
        this->updateDirection(1);  // Bootstrap: exercises shifts without a fit.
    }
};

template<typename T>
class AsyncExecutionTest : public testing::Test {};
using Precisions = testing::Types<float, double>;
TYPED_TEST_SUITE(AsyncExecutionTest, Precisions);

TYPED_TEST(AsyncExecutionTest, backupRestoreAndResidualChain) {
    using T = TypeParam;
    ScenarioTree<T> tree(SPOCK_TEST_DATA_DIR);
    ProblemData<T> data(tree);
    ExecutionTestCache<T> cache(tree, data);
    const size_t n = cache.iterate().numEl();
    std::vector<T> a(n), b(n);
    for (size_t i = 0; i < n; ++i) { a[i] = T(i % 11); b[i] = T(i % 7); }
    cache.iterate().upload(a);
    cache.candidate().upload(b);
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    cache.backup();
    cache.acceptCandidate();
    cache.iterate() *= T(2);
    cache.iterate().deviceCopyToAsync(cache.candidate());
    cache.restore();
    cache.computeResidual();
    std::vector<T> actual;
    cache.residual().download(actual);  // The only host completion boundary.
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(actual[i], b[i]);
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
}

TYPED_TEST(AsyncExecutionTest, borrowedHistoryColumnsStayOrdered) {
    using T = TypeParam;
    ScenarioTree<T> tree(SPOCK_TEST_DATA_DIR);
    ProblemData<T> data(tree);
    ExecutionTestCache<T> cache(tree, data);
    const size_t rows = cache.iterate().numEl(), columns = cache.historyC().numCols();
    std::vector<T> c(rows * columns), y(rows * columns), delta(rows), residual(rows);
    for (size_t i = 0; i < c.size(); ++i) { c[i] = T(i % 23); y[i] = T(i % 19); }
    for (size_t i = 0; i < rows; ++i) { delta[i] = T(i % 13); residual[i] = T(i % 5); }
    cache.historyC().upload(c); cache.historyY().upload(y);
    cache.deltaIterate().upload(delta); cache.deltaResidual().upload(residual);
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    cache.shiftHistory();
    std::vector<T> actualC, actualY;
    cache.historyC().download(actualC); cache.historyY().download(actualY);
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_EQ(actualC[i], delta[i] - residual[i]);
        EXPECT_EQ(actualY[i], residual[i]);
    }
    for (size_t i = rows; i < c.size(); ++i) {
        EXPECT_EQ(actualC[i], c[i - rows]);
        EXPECT_EQ(actualY[i], y[i - rows]);
    }
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
}

TYPED_TEST(AsyncExecutionTest, differentStreamsAreRejected) {
    using T = TypeParam;
    DTensor<T> source(8), destination(8);
    destination.setStreamIdx(1);
    EXPECT_THROW(source.deviceCopyToAsync(destination), std::invalid_argument);
    EXPECT_THROW(memCpyNode2Node(destination, source, 0, 0, 8), std::invalid_argument);
}

TYPED_TEST(AsyncExecutionTest, andersonFitAndFallbackOnSecondStream) {
    using T = TypeParam;
    constexpr size_t rows = 64, columns = 3;
    std::vector<T> history(rows * columns), cValues(rows * columns), values(rows);
    for (size_t col = 0; col < columns; ++col) {
        history[col * rows + col] = T(1);
        cValues[col * rows + col] = T(.5);
        values[col] = T(col + 1);
    }
    for (bool incremental : {false, true}) {
        DTensor<T> c(rows, columns), y(rows, columns), q(rows, columns);
        DTensor<T> work(rows), gamma(columns), residual(rows), direction(rows);
        for (auto *tensor : {&c, &y, &q, &work, &gamma, &residual, &direction})
            tensor->setStreamIdx(1);
        c.upload(cValues); y.upload(history); residual.upload(values);
        AndersonOptions options;
        options.regularization = .125;
        options.qr.incremental = incremental;
        RegularisedAnderson<T> fit(c, y, q, work, gamma, options);
        const auto allocated = Session::getInstance().totalAllocatedBytes();
        ASSERT_TRUE(fit.compute(residual, direction).accepted());
        std::vector<T> result;
        direction.download(result);
        const double tolerance = std::is_same_v<T, float> ? 2e-5 : 2e-11;
        for (size_t i = 0; i < rows; ++i)
            EXPECT_NEAR(result[i], -values[i] * (1 + .5 / 1.125), tolerance);
        // A zero history takes the residual fallback, including the coefficient
        // upload and subsequent GPU copy/scale on the selected stream.
        y.upload(std::vector<T>(rows * columns));
        fit.reset();
        EXPECT_EQ(fit.compute(residual, direction).status, AndersonFitStatus::rankZero);
        direction.download(result);
        for (size_t i = 0; i < rows; ++i) EXPECT_EQ(result[i], -values[i]);
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
}

}  // namespace
