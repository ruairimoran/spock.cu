#include <gtest/gtest.h>
#include "cache.cuh"

namespace {

__global__ void delayProjection(unsigned long long cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {}
}

template<typename T>
__global__ void produceIterate(T *values, size_t size, int seed, unsigned long long cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {}
    for (size_t i = threadIdx.x; i < size; i += blockDim.x)
        values[i] = T(int((i + seed) % 23) - 11) / T(16);
}

template<typename T>
class ProjectionTestCache : public Cache<T> {
public:
    ProjectionTestCache(ScenarioTree<T> &tree, ProblemData<T> &data, spock::ProjectionMode mode)
        : Cache<T>(tree, data, T(1e-4), T(0), T(0), 2000, 8, 3, false, {}, mode) {}
    using Cache<T>::iter;
    using Cache<T>::initialiseState;
    DTensor<T> &candidate() { return *this->m_d_iterateCandidate; }
    DTensor<T> &workspace() { return *this->m_d_workIterate; }
    void checkMainViews() {
        for (const auto *view : {this->m_d_y.get(), this->m_d_t.get(), this->m_d_s.get(),
                                 this->m_d_workYTS.get(), this->m_d_projectedYTS.get()})
            EXPECT_EQ(view->streamIdx(), 0u);
        EXPECT_EQ(this->m_data.risk()->nullspaceProj().streamIdx(), 0u);
    }
};

// ProblemData contains mutable operator workspaces and cost views. Each cache
// gets its own problem, rather than letting one cache overwrite another's views.
template<typename T>
struct ProjectionFixture {
    ScenarioTree<T> tree{SPOCK_TEST_DATA_DIR};
    ProblemData<T> data{tree};
    ProjectionTestCache<T> cache;
    explicit ProjectionFixture(spock::ProjectionMode mode) : cache(tree, data, mode) {}
};

template<typename T>
class ConcurrentProjectionTest : public testing::Test {};
using Precisions = testing::Types<float, double>;
TYPED_TEST_SUITE(ConcurrentProjectionTest, Precisions);

TYPED_TEST(ConcurrentProjectionTest, cpMapMatchesWithDelayedProducerAndConsumer) {
    using T = TypeParam;
    ProjectionFixture<T> sequential(spock::ProjectionMode::sequential);
    ProjectionFixture<T> concurrent(spock::ProjectionMode::concurrent);
    ASSERT_EQ(concurrent.cache.projectionMode(), spock::ProjectionMode::concurrent);
    const size_t size = sequential.cache.solution().numEl();
    std::vector<T> root(sequential.tree.numStates(), T(.1));
    sequential.cache.initialiseState(root); concurrent.cache.initialiseState(root);
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    cudaStream_t aux;
    gpuErrChk(cublasGetStream(Session::getInstance().cuBlasHandle(1), &aux));
    for (int sample = 0; sample < 20; ++sample) {
        std::vector<T> values(size);
        for (size_t i = 0; i < size; ++i) values[i] = T(int((i + sample) % 23) - 11) / T(16);
        sequential.cache.solution().upload(values);
        sequential.cache.iter();
        std::vector<T> expected, actual;
        sequential.cache.candidate().download(expected);

        // The fork must wait for a queued main-stream producer. The join must
        // wait for the deliberately delayed auxiliary projection before copying
        // the complete candidate or allowing the caller's download to finish.
        const auto main = spock::executionStream(concurrent.cache.solution());
        delayProjection<<<1, 32, 0, aux>>>(sample % 2 ? 3000000 : 0);
        produceIterate<<<1, 256, 0, main>>>(concurrent.cache.solution().raw(), size, sample, sample % 2 ? 0 : 3000000);
        gpuErrChk(cudaGetLastError());
        concurrent.cache.iter();
        concurrent.cache.candidate().download(actual);
        ASSERT_EQ(actual.size(), expected.size());
        const double tolerance = std::is_same_v<T, float> ? 5e-5 : 2e-11;
        for (size_t i = 0; i < size; ++i)
            ASSERT_NEAR(actual[i], expected[i], tolerance * (1 + std::abs(expected[i])))
                << "sample " << sample << ", entry " << i;
        concurrent.cache.checkMainViews();
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
}

TYPED_TEST(ConcurrentProjectionTest, residentColdAndWarmSolves) {
    using T = TypeParam;
    ProjectionFixture<T> sequential(spock::ProjectionMode::sequential);
    ProjectionFixture<T> concurrent(spock::ProjectionMode::concurrent);
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    const auto pointer = concurrent.cache.solution().raw();
    for (int step = 0; step < 3; ++step) {
        std::vector<T> state(sequential.tree.numStates(), T(.05 * (step + 1)));
        ASSERT_EQ(sequential.cache.runSpockResident(state, step > 0), converged);
        ASSERT_EQ(concurrent.cache.runSpockResident(state, step > 0), converged);
        const auto reference = sequential.cache.rootInput(), actual = concurrent.cache.rootInput();
        ASSERT_EQ(actual.size(), reference.size());
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], reference[i], 2e-3);
        EXPECT_EQ(concurrent.cache.solution().raw(), pointer);
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
}

TYPED_TEST(ConcurrentProjectionTest, destructionDrainsPendingWork) {
    using T = TypeParam;
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    {
        ProjectionFixture<T> concurrent(spock::ProjectionMode::concurrent);
        std::vector<T> root(concurrent.tree.numStates(), T(.1));
        concurrent.cache.initialiseState(root);
        for (int i = 0; i < 10; ++i) concurrent.cache.iter();
        // No download or explicit wait: cache destruction must join the work
        // before it releases any backing allocations or projection events.
    }
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
}

TEST(ProjectionConfigurationTest, singleStreamFallback) {
    if (Session::getInstance().numStreams() != 1) GTEST_SKIP() << "Separate single-stream CTest entry";
    ScenarioTree<double> tree(SPOCK_TEST_DATA_DIR);
    ProblemData<double> data(tree);
    {
        auto cache = CacheBuilder<double>(tree, data).make_unique();
        EXPECT_EQ(cache->projectionMode(), spock::ProjectionMode::sequential);
    }
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    EXPECT_THROW(CacheBuilder<double>(tree, data)
                     .projectionMode(spock::ProjectionMode::concurrent).make_unique(), std::invalid_argument);
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
}

}  // namespace
