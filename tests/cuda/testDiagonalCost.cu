#include <gtest/gtest.h>
#include "cache.cuh"
#include <numeric>

template<typename T>
struct CostResults {
    std::vector<T> forward, x, u, scalar, translation, projection;
    size_t storage = 0;
};

template<typename T>
CostResults<T> evaluateCost(const std::string &path, bool linear, bool leaf) {
    ScenarioTree<T> tree(path);
    const size_t before = Session::getInstance().totalAllocatedBytes();
    std::unique_ptr<Cost<T>> cost;
    if (linear) cost = std::make_unique<CostQuadraticPlusLinear<T>>(tree, leaf ? ::leaf : nonleaf);
    else cost = std::make_unique<CostQuadratic<T>>(tree, leaf ? ::leaf : nonleaf);
    CostResults<T> result;
    result.storage = Session::getInstance().totalAllocatedBytes() - before;
    const size_t nx = tree.numStates(), nu = tree.numInputs();
    const size_t nodes = leaf ? tree.numLeafNodes() : tree.numNodes(), dim = cost->dimPerNode();
    const size_t controls = leaf ? 0 : tree.numNonleafNodes();
    auto values = [](size_t n, int shift) {
        std::vector<T> data(n);
        for (size_t i = 0; i < n; ++i) data[i] = T(int((i + shift) % 17) - 8) / T(10);
        return data;
    };
    auto x = values(nx * nodes, 0), u = values(nu * controls, 3), scalar = values(nodes, 4);
    auto dual = values(dim * nodes, 5);
    if (!leaf) std::fill(dual.begin(), dual.begin() + dim, T(0));
    DTensor<T> dx(x, nx, 1, nodes), du;
    if (!leaf) du = DTensor<T>(u, nu, 1, controls);
    DTensor<T> ds(scalar, 1, 1, nodes), output(dim, 1, nodes, true), dd(dual, dim, 1, nodes);
    DTensor<T> ax(nx, 1, nodes), au;
    if (!leaf) au = DTensor<T>(nu, 1, controls);
    DTensor<T> as(1, 1, nodes, true), work(dim * nodes);
    cost->reshape(work, 0, dim * nodes - 1);
    work.reshape(dim, 1, nodes);
    const size_t allocated = Session::getInstance().totalAllocatedBytes();
    for (int repeat = 0; repeat < 3; ++repeat) {
        ds.upload(scalar);
        if (leaf) cost->op(output, dx, ds);
        else cost->op(output, dx, du, ds);
        ax.upload(std::vector<T>(nx * nodes, T(.25)));
        if (!leaf) au.upload(std::vector<T>(nu * controls, T(.25)));
        as.upload(std::vector<T>(nodes, T(0)));
        if (leaf) cost->adj(dd, ax, as);
        else cost->adj(dd, ax, au, as);
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
    output.download(result.forward); ax.download(result.x); as.download(result.scalar);
    if (!leaf) au.download(result.u);
    double lhs = 0, rhs = 0;
    for (size_t i = 0; i < dual.size(); ++i) lhs += double(dual[i]) * result.forward[i];
    for (size_t i = 0; i < x.size(); ++i) rhs += double(x[i]) * (result.x[i] - T(.25));
    for (size_t i = 0; i < u.size(); ++i) rhs += double(u[i]) * (result.u[i] - T(.25));
    for (size_t i = 0; i < scalar.size(); ++i) rhs += double(scalar[i]) * result.scalar[i];
    EXPECT_NEAR(lhs, rhs, (std::is_same_v<T, float> ? 2e-5 : 1e-11));
    work.upload(values(dim * nodes, 2));
    cost->translate(work); work.download(result.translation);
    cost->project(); work.download(result.projection);
    gpuErrChk(cudaDeviceSynchronize());
    return result;
}

template<typename T>
void checkCompactCosts(const std::string &name, bool linear) {
    const std::string base = std::string(SPOCK_TEST_FIXTURE_DIR) + name + "/";
    for (bool leaf : {false, true}) {
        SCOPED_TRACE(name + (leaf ? " leaf" : " stage"));
        const auto compact = evaluateCost<T>(base + "compact/data/", linear, leaf);
        const auto dense = evaluateCost<T>(base + "dense/data/", linear, leaf);
        auto compare = [](const auto &actual, const auto &expected) {
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], expected[i], (std::is_same_v<T, float> ? 2e-5 : 1e-11)) << i;
        };
        compare(compact.forward, dense.forward); compare(compact.x, dense.x);
        compare(compact.u, dense.u); compare(compact.scalar, dense.scalar);
        compare(compact.translation, dense.translation); compare(compact.projection, dense.projection);
        // mixed_r has a dense terminal Q, so its terminal storage is identical.
        if (!(leaf && name.find("mixed_r") != std::string::npos)) EXPECT_LT(compact.storage, dense.storage);
    }
}

template<typename T>
void checkAllCompactCosts() {
    for (bool linear : {false, true})
        for (const std::string mode : {"diagonal", "mixed_q", "mixed_r"})
            checkCompactCosts<T>((linear ? "quadratic_plus_linear_" : "quadratic_") + mode, linear);
    checkCompactCosts<T>("scalar_quadratic", false);
}

TEST(DiagonalCostTest, operatorsFloat) { checkAllCompactCosts<float>(); }
TEST(DiagonalCostTest, operatorsDouble) { checkAllCompactCosts<double>(); }

TEST(DiagonalCostTest, solverMatchesDense) {
    const std::string base = std::string(SPOCK_TEST_FIXTURE_DIR) + "quadratic_mixed_q/";
    std::vector<std::vector<double>> controls;
    for (const std::string representation : {"dense", "compact"}) {
        ScenarioTree<double> tree(base + representation + "/data/");
        ProblemData<double> problem(tree);
        auto cache = CacheBuilder<double>(tree, problem).tol(1e-5).maxIters(3000).make_unique();
        std::vector<double> state(tree.numStates(), .2);
        ASSERT_EQ(cache->runSpockResident(state, false), converged);
        controls.push_back(cache->rootInput());
        ASSERT_EQ(cache->runSpockResident(state, true), converged);
    }
    ASSERT_EQ(controls[0].size(), controls[1].size());
    for (size_t i = 0; i < controls[0].size(); ++i) EXPECT_NEAR(controls[0][i], controls[1][i], 5e-5);
}
