#include <gtest/gtest.h>
#include <limits>
#include "cache.cuh"
#include <array>

namespace {

template<typename T>
class GraphTestCache : public Cache<T> {
public:
    GraphTestCache(ScenarioTree<T> &tree, ProblemData<T> &data, spock::GraphMode mode, bool debug = false)
        : Cache<T>(tree, data, T(1e-4), T(0), T(0), 2000, 8, 3, debug, {},
                   spock::ProjectionMode::sequential, mode) {}
    using Cache<T>::iter;
    using Cache<T>::initialiseState;
    DTensor<T> &candidate() { return *this->m_d_iterateCandidate; }
    size_t linearOperatorCalls() const { return this->m_callsToL; }
};

template<typename T>
struct GraphFixture {
    ScenarioTree<T> tree;
    ProblemData<T> data;
    GraphTestCache<T> cache;
    GraphFixture(const std::string &path, spock::GraphMode mode, bool debug = false)
        : tree(path), data(tree), cache(tree, data, mode, debug) {}
};

template<typename T>
void compareGraphMaps(const std::string &path) {
    GraphFixture<T> reference(path, spock::GraphMode::none, true);
    std::array<std::unique_ptr<GraphFixture<T>>, 2> graphs;
    graphs[0] = std::make_unique<GraphFixture<T>>(path, spock::GraphMode::dynamics, true);
    graphs[1] = std::make_unique<GraphFixture<T>>(path, spock::GraphMode::cp, true);
    const size_t size = reference.cache.solution().numEl();
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    for (int sample = 0; sample < 6; ++sample) {
        std::vector<T> input(size), root(reference.tree.numStates(), T(.1 * (sample + 1)));
        for (size_t i = 0; i < size; ++i) input[i] = T(int((i + sample) % 23) - 11) / T(16);
        std::vector<T> expected;
        auto evaluate = [&](auto &fixture, std::vector<T> &result) {
            auto &cache = fixture.cache;
            cache.solution().upload(input);
            cache.initialiseState(root);
            if (sample % 2) {
                // Exercise the public L-only path between graph replays.
                // Host owner swaps in dense costs would invalidate fixed roles.
                DTensor<T> primal(cache.solution(), 0, 0, cache.sizePrimal() - 1);
                auto dual = cache.getDual(primal);
            }
            const auto calls = cache.linearOperatorCalls();
            cache.iter();
            EXPECT_EQ(cache.linearOperatorCalls(), calls + 2);
            EXPECT_EQ(cache.operatorEvaluations(), size_t(sample + 1));
            cache.candidate().download(result);
        };
        evaluate(reference, expected);
        for (auto &graph : graphs) {
            std::vector<T> actual;
            evaluate(*graph, actual);
            EXPECT_EQ(graph->cache.graphStatistics().launches, size_t(sample + 1));
            EXPECT_GT(graph->cache.graphStatistics().nodes, 0u);
            ASSERT_EQ(actual.size(), expected.size());
            const double tolerance = std::is_same_v<T, float> ? 5e-5 : 2e-11;
            for (size_t i = 0; i < size; ++i)
                ASSERT_NEAR(actual[i], expected[i], tolerance * (1 + std::abs(expected[i])))
                    << "sample " << sample << ", element " << i;
        }
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
}

template<typename T>
class CudaGraphTest : public testing::Test {};
using Precisions = testing::Types<float, double>;
TYPED_TEST_SUITE(CudaGraphTest, Precisions);

TYPED_TEST(CudaGraphTest, changedInputsAndHostBookkeeping) {
    compareGraphMaps<TypeParam>(SPOCK_TEST_DATA_DIR);
}

TYPED_TEST(CudaGraphTest, denseAndMixedCostWorkspaces) {
    const std::string base = SPOCK_TEST_FIXTURE_DIR;
    for (const char *name : {"quadratic/data/", "quadratic_plus_linear/data/",
                             "quadratic_mixed_q/compact/data/", "quadratic_mixed_r/compact/data/",
                             "quadratic_diagonal/compact/data/"}) {
        SCOPED_TRACE(name);
        compareGraphMaps<TypeParam>(base + name);
    }
}

TYPED_TEST(CudaGraphTest, coldWarmAndPendingDestruction) {
    using T = TypeParam;
    const auto before = Session::getInstance().totalAllocatedBytes();
    for (auto mode : {spock::GraphMode::dynamics, spock::GraphMode::cp}) {
        GraphFixture<T> fixture(SPOCK_TEST_DATA_DIR, mode);
        const auto allocated = Session::getInstance().totalAllocatedBytes();
        const auto pointer = fixture.cache.solution().raw();
        for (int step = 0; step < 3; ++step) {
            std::vector<T> state(fixture.tree.numStates(), T(.05 * (step + 1)));
            ASSERT_EQ(fixture.cache.runSpockResident(state, step > 0), converged);
            EXPECT_EQ(fixture.cache.graphStatistics().launches, fixture.cache.operatorEvaluations());
            EXPECT_EQ(fixture.cache.solution().raw(), pointer);
            EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
        }
        // No explicit wait or download before the graph-owning cache dies.
        for (int i = 0; i < 10; ++i) fixture.cache.iter();
    }
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), before);
}

template<typename T>
void checkGraphDynamics(const std::string &name, bool affine, bool update,
                        size_t blockSize = 0, size_t backwardWarps = 4,
                        const std::string &replacement = "linear_update") {
    const std::string base = std::string(SPOCK_TEST_FIXTURE_DIR) + "dynamics_projection/";
    const std::string path = base + name + "/data/";
    ScenarioTree<T> tree(path);
    std::unique_ptr<Dynamics<T>> projection;
    if (affine) projection = std::make_unique<Affine<T>>(tree);
    else projection = std::make_unique<Linear<T>>(tree);
    projection->configurePartialLifting(blockSize, backwardWarps);
    if (blockSize) EXPECT_EQ(projection->partialLiftingStatistics().backwardWarps, backwardWarps);
    DTensor<T> root(tree.numStates(), 1, 1, true);
    DTensor<T> states(tree.numStates(), 1, tree.numNodes(), true);
    DTensor<T> inputs(tree.numInputs(), 1, tree.numNonleafNodes(), true);
    projection->project(root, states, inputs);
    spock::detail::CudaGraph graph;
    graph.capture(spock::executionStream(states), [&] { projection->project(root, states, inputs); });
    if (blockSize)
        EXPECT_THROW(projection->configurePartialLifting(blockSize, backwardWarps == 4 ? 8 : 4),
                     std::logic_error);
    for (int version = 0; version < (update ? 2 : 1); ++version) {
        const auto current = version ? base + replacement + "/data/" : path;
        if (version) {
            const auto complete = projection->projectionTensors();
            for (size_t index : {size_t(5),size_t(6)})
                complete[index]->upload(std::vector<T>(complete[index]->numEl(),std::numeric_limits<T>::quiet_NaN()));
            const auto factors = projection->linearUpdateTensors();
            const auto &names = Dynamics<T>::linearUpdateNames;
            for (size_t i = 0; i < factors.size(); ++i) {
                auto factor = DTensor<T>::parseFromFile(current + "dynamics_" + names[i] + tree.fpFileExt());
                factor.deviceCopyTo(*factors[i]);
            }
            const size_t allocated = Session::getInstance().totalAllocatedBytes();
            projection->refreshProjectionFactors();
            EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
        }
        for (int sample = 0; sample < 3; ++sample) {
            const auto prefix = current + "projection_" + std::to_string(sample) + "_";
            auto load = [&](const char *suffix) { return DTensor<T>::parseFromFile(prefix + suffix + tree.fpFileExt()); };
            auto xbar = load("xbar"), ubar = load("ubar"), initial = load("root");
            xbar.deviceCopyToAsync(states); ubar.deviceCopyToAsync(inputs); initial.deviceCopyToAsync(root);
            graph.launch();
            auto compare = [&](DTensor<T> &actual, const char *suffix) {
                std::vector<T> result, reference;
                actual.download(result); load(suffix).download(reference);
                ASSERT_EQ(result.size(), reference.size());
                const double tolerance = std::is_same_v<T, float> ? 3e-4 : 2e-10;
                for (size_t i = 0; i < result.size(); ++i)
                    ASSERT_NEAR(result[i], reference[i], tolerance * (1 + std::abs(reference[i]))) << name << ", element " << i;
            };
            compare(states, "x"); compare(inputs, "u");
        }
    }
}

TYPED_TEST(CudaGraphTest, projectionReferencesAndResidentFactorUpdates) {
    using T = TypeParam;
    checkGraphDynamics<T>("linear", false, true);
    checkGraphDynamics<T>("affine", true, false);
    checkGraphDynamics<T>("chain", false, false);
    checkGraphDynamics<T>("scalar", true, false);
    checkGraphDynamics<T>("wide", false, false);
    checkGraphDynamics<T>("scaled", true, false);
    checkGraphDynamics<T>("tiled", true, false);
}

TEST(CudaGraphConfigurationTest, sequentialRequiredAndCaptureCleanup) {
    ScenarioTree<double> tree(SPOCK_TEST_DATA_DIR);
    ProblemData<double> data(tree);
    auto cache = CacheBuilder<double>(tree, data).cudaGraphs(spock::GraphMode::cp).make_unique();
    EXPECT_EQ(cache->projectionMode(), spock::ProjectionMode::sequential);
    EXPECT_THROW(CacheBuilder<double>(tree, data).projectionMode(spock::ProjectionMode::concurrent)
                     .cudaGraphs(spock::GraphMode::cp).make_unique(), std::invalid_argument);
    EXPECT_THROW(CacheBuilder<double>(tree, data).cudaGraphs(static_cast<spock::GraphMode>(99)).make_unique(),
                 std::invalid_argument);
    DTensor<double> scalar(1, 1, 1, true);
    spock::detail::CudaGraph graph;
    const auto stream = spock::executionStream(scalar);
    EXPECT_THROW(graph.capture(stream, [] { throw std::runtime_error("Interrupted capture"); }), std::runtime_error);
    cudaStreamCaptureStatus status;
    gpuErrChk(cudaStreamIsCapturing(stream, &status));
    EXPECT_EQ(status, cudaStreamCaptureStatusNone);
    scalar.upload(std::vector<double>{2.});
    graph.capture(stream, [&] { scalar *= 2.; });
    graph.launch();
    EXPECT_EQ(scalar(0), 4.);
}

}  // namespace

TYPED_TEST(CudaGraphTest, partialLiftingReferencesAndResidentUpdates) {
    using T = TypeParam;
    for (size_t backwardWarps : {size_t(4), size_t(8)}) {
        SCOPED_TRACE("backward warps " + std::to_string(backwardWarps));
        for (size_t blockSize : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(8)}) {
            SCOPED_TRACE("block size " + std::to_string(blockSize));
            checkGraphDynamics<T>("chain", false, true, blockSize, backwardWarps, "chain_update");
            checkGraphDynamics<T>("affine_tail", true, false, blockSize, backwardWarps);
            checkGraphDynamics<T>("tail_tiled", true, false, blockSize, backwardWarps);
            checkGraphDynamics<T>("affine", true, false, blockSize, backwardWarps);
        }
    }
}
