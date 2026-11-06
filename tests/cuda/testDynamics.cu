#include <gtest/gtest.h>
#include "dynamics.cuh"
#include <cmath>
#include <limits>
#include <type_traits>

template<typename T>
class DynamicsProjectionTest : public testing::Test {};

using ProjectionPrecisions = testing::Types<float, double>;
TYPED_TEST_SUITE(DynamicsProjectionTest, ProjectionPrecisions);

template<typename T>
void expectProjectionNear(DTensor<T> &actual, DTensor<T> &expected) {
    std::vector<T> values, reference;
    actual.download(values);
    expected.download(reference);
    ASSERT_EQ(values.size(), reference.size());
    const double tolerance = std::is_same_v<T, float> ? 3e-4 : 2e-10;
    for (size_t i = 0; i < values.size(); ++i)
        ASSERT_NEAR(values[i], reference[i], tolerance * (1. + std::abs(reference[i]))) << "element " << i;
}

template<typename T>
void checkProjectionReferences(Dynamics<T> &projection, ScenarioTree<T> &tree, const std::string &path) {
    const std::string ext = tree.fpFileExt();
    DTensor<T> states(tree.numStates(), 1, tree.numNodes());
    DTensor<T> inputs(tree.numInputs(), 1, tree.numNonleafNodes());
    for (int trial = 0; trial < 3; ++trial) {
        SCOPED_TRACE("trial " + std::to_string(trial));
        const std::string prefix = path + "projection_" + std::to_string(trial) + "_";
        auto originalStates = DTensor<T>::parseFromFile(prefix + "xbar" + ext);
        auto originalInputs = DTensor<T>::parseFromFile(prefix + "ubar" + ext);
        auto root = DTensor<T>::parseFromFile(prefix + "root" + ext);
        auto expectedStates = DTensor<T>::parseFromFile(prefix + "x" + ext);
        auto expectedInputs = DTensor<T>::parseFromFile(prefix + "u" + ext);
        originalStates.deviceCopyTo(states);
        originalInputs.deviceCopyTo(inputs);
        const size_t allocated = Session::getInstance().totalAllocatedBytes();
        projection.project(root, states, inputs);
        expectProjectionNear(states, expectedStates);
        expectProjectionNear(inputs, expectedInputs);
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);

        // A projection is idempotent. Reusing its workspaces must not accumulate
        // affine offsets or retain right-hand sides from an earlier call.
        projection.project(root, states, inputs);
        expectProjectionNear(states, expectedStates);
        expectProjectionNear(inputs, expectedInputs);
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
}

template<typename T>
void checkDynamicsFixture(const std::string &name, bool affine) {
    const std::string path = std::string(SPOCK_TEST_FIXTURE_DIR) + "dynamics_projection/" + name + "/data/";
    ScenarioTree<T> tree(path);
    std::unique_ptr<Dynamics<T>> projection;
    if (affine) projection = std::make_unique<Affine<T>>(tree);
    else projection = std::make_unique<Linear<T>>(tree);
    checkProjectionReferences(*projection, tree, path);
}

TYPED_TEST(DynamicsProjectionTest, denseLinearIrregularTree) { checkDynamicsFixture<TypeParam>("linear", false); }
TYPED_TEST(DynamicsProjectionTest, denseAffineIrregularTree) { checkDynamicsFixture<TypeParam>("affine", true); }
TYPED_TEST(DynamicsProjectionTest, twelveStageChain) { checkDynamicsFixture<TypeParam>("chain", false); }
TYPED_TEST(DynamicsProjectionTest, singleStageScalar) { checkDynamicsFixture<TypeParam>("scalar", true); }
TYPED_TEST(DynamicsProjectionTest, fiftyRankDeficientInputs) { checkDynamicsFixture<TypeParam>("wide", false); }
TYPED_TEST(DynamicsProjectionTest, scaledAffine) { checkDynamicsFixture<TypeParam>("scaled", true); }
TYPED_TEST(DynamicsProjectionTest, affineAcrossProductTiles) { checkDynamicsFixture<TypeParam>("tiled", true); }

TYPED_TEST(DynamicsProjectionTest, residentFactorUpdates) {
    using T = TypeParam;
    const std::string base = std::string(SPOCK_TEST_FIXTURE_DIR) + "dynamics_projection/";
    ScenarioTree<T> tree(base + "linear/data/");
    Linear<T> projection(tree);
    checkProjectionReferences(projection, tree, base + "linear/data/");
    // Obsolete resident values must never influence the linear projection.
    const auto complete = projection.projectionTensors();
    for (size_t index : {size_t(5),size_t(6)})
        complete[index]->upload(std::vector<T>(complete[index]->numEl(),std::numeric_limits<T>::quiet_NaN()));
    const auto tensors = projection.linearUpdateTensors();
    const auto &names = Dynamics<T>::linearUpdateNames;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto replacement = DTensor<T>::parseFromFile(base + "linear_update/data/dynamics_" + std::string(names[i]) + tree.fpFileExt());
        replacement.deviceCopyTo(*tensors[i]);
    }
    // The solver's pointer arrays must continue to refer to the resident
    // factors after their values change between MPC optimization problems.
    checkProjectionReferences(projection, tree, base + "linear_update/data/");
}

TYPED_TEST(DynamicsProjectionTest, linearUpdateLayoutContainsOnlyActiveFactors) {
    using T = TypeParam;
    const auto path = std::string(SPOCK_TEST_FIXTURE_DIR) + "dynamics_projection/linear/data/";
    ScenarioTree<T> tree(path);
    Linear<T> projection(tree);
    const auto complete = projection.projectionTensors();
    const auto active = projection.linearUpdateTensors();
    ASSERT_EQ(active.size(),6);
    const std::array<size_t,6> indices = {0,1,2,3,4,7};
    const std::array<const char *,6> names = {"BTr","AB","K","KTr","ABKTr","lowCholesky"};
    for (size_t i = 0; i < active.size(); ++i) {
        EXPECT_EQ(active[i],complete[indices[i]]);
        EXPECT_STREQ(names[i],Dynamics<T>::linearUpdateNames[i]);
    }
}

TYPED_TEST(DynamicsProjectionTest, affineOnlineUpdatesRemainUnsupported) {
    using T = TypeParam;
    const auto path = std::string(SPOCK_TEST_FIXTURE_DIR) + "dynamics_projection/affine/data/";
    ScenarioTree<T> tree(path);
    Affine<T> projection(tree);
    EXPECT_THROW(projection.linearUpdateTensors(),std::invalid_argument);
}
