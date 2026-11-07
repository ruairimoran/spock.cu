#include <gtest/gtest.h>
#include "cache.cuh"

template<typename T>
void checkDenseCost(bool linear) {
    const std::string path = std::string(SPOCK_TEST_FIXTURE_DIR) +
        (linear ? "quadratic_plus_linear/data/" : "quadratic/data/");
    ScenarioTree<T> tree(path);
    std::unique_ptr<Cost<T>> cost;
    if (linear) cost = std::make_unique<CostQuadraticPlusLinear<T>>(tree);
    else cost = std::make_unique<CostQuadratic<T>>(tree);
    const size_t nx = tree.numStates(), nu = tree.numInputs(), nodes = tree.numNodes();
    const size_t nonleaf = tree.numNonleafNodes(), dim = nx + nu + 2;
    std::vector<T> x(nx * nodes), u(nu * nonleaf), t(nodes), dual(dim * nodes);
    for (size_t i = 0; i < x.size(); ++i) x[i] = T(int(i % 17) - 8) / T(10);
    for (size_t i = 0; i < u.size(); ++i) u[i] = T(int(i % 13) - 6) / T(10);
    for (size_t i = 0; i < t.size(); ++i) t[i] = T(i + 1) / T(5);
    for (size_t i = 0; i < dual.size(); ++i) dual[i] = T(int(i % 19) - 9) / T(10);
    std::vector<size_t> ancestors; tree.d_ancestors().download(ancestors);
    auto load = [&](const char *name) {
        std::vector<T> values;
        DTensor<T>::parseFromFile(path + "nonleafCost_" + name + tree.fpFileExt()).download(values);
        return values;
    };
    const auto Q = load("sqrtQ"), R = load("sqrtR");
    const auto q = linear ? load("q") : std::vector<T>(nx * nodes);
    const auto r = linear ? load("r") : std::vector<T>(nu * nodes);
    std::vector<T> expectedOp(dim * nodes), expectedX(nx * nodes), expectedU(nu * nonleaf), expectedT(nodes);
    // Independent host matrix-vector products, including the linear terms.
    for (size_t node = 1; node < nodes; ++node) {
        const size_t parent = ancestors[node];
        double scalar = t[node];
        const double sum = double(dual[node * dim + nx + nu]) + dual[node * dim + nx + nu + 1];
        for (size_t row = 0; row < nx; ++row) {
            double op = 0, adj = 0;
            for (size_t col = 0; col < nx; ++col) {
                op += double(Q[node * nx * nx + col * nx + row]) * x[parent * nx + col];
                adj += double(Q[node * nx * nx + row * nx + col]) * dual[node * dim + col];
            }
            expectedOp[node * dim + row] = T(op);
            expectedX[parent * nx + row] += T(adj - 0.5 * q[node * nx + row] * sum);
            scalar -= double(q[node * nx + row]) * x[parent * nx + row];
        }
        for (size_t row = 0; row < nu; ++row) {
            double op = 0, adj = 0;
            for (size_t col = 0; col < nu; ++col) {
                op += double(R[node * nu * nu + col * nu + row]) * u[parent * nu + col];
                adj += double(R[node * nu * nu + row * nu + col]) * dual[node * dim + nx + col];
            }
            expectedOp[node * dim + nx + row] = T(op);
            expectedU[parent * nu + row] += T(adj - 0.5 * r[node * nu + row] * sum);
            scalar -= double(r[node * nu + row]) * u[parent * nu + row];
        }
        expectedOp[node * dim + nx + nu] = expectedOp[node * dim + nx + nu + 1] = T(0.5 * scalar);
        expectedT[node] = T(0.5 * sum);
    }
    DTensor<T> dx(x, nx, 1, nodes), du(u, nu, 1, nonleaf), dt(t, 1, 1, nodes);
    DTensor<T> output(dim, 1, nodes, true), ddual(dual, dim, 1, nodes);
    DTensor<T> adjX(nx, 1, nodes), adjU(nu, 1, nonleaf), adjT(1, 1, nodes);
    const size_t allocated = Session::getInstance().totalAllocatedBytes();
    auto compare = [](const std::vector<T> &actual, const std::vector<T> &expected) {
        ASSERT_EQ(actual.size(), expected.size());
        const double tolerance = std::is_same_v<T, float> ? 2e-5 : 1e-11;
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "entry " << i;
    };
    for (int repeat = 0; repeat < 3; ++repeat) {
        dt.upload(t);
        cost->op(output, dx, du, dt);
        std::vector<T> actual; output.download(actual); compare(actual, expectedOp);
        adjX.upload(std::vector<T>(nx * nodes)); adjU.upload(std::vector<T>(nu * nonleaf));
        adjT.upload(std::vector<T>(nodes));
        cost->adj(ddual, adjX, adjU, adjT);
        adjX.download(actual); compare(actual, expectedX);
        adjU.download(actual); compare(actual, expectedU);
        adjT.download(actual); compare(actual, expectedT);
        EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), allocated);
    }
    // An unconstrained problem needs empty tensor placeholders, not invalid
    // zero-length allocations or null references passed into the operator.
    ProblemData<T> problem(tree);
    CacheBuilder<T> builder(tree, problem);
    auto cache = builder.maxIters(1500).tol(T(1e-3)).make_unique();
    if (!linear) {
        std::vector<T> initial(nx, T(0));
        EXPECT_EQ(cache->runSpock(initial), converged);
    }
}

TEST(DenseCostTest, quadraticFloat) { checkDenseCost<float>(false); }
TEST(DenseCostTest, quadraticDouble) { checkDenseCost<double>(false); }
TEST(DenseCostTest, quadraticPlusLinearFloat) { checkDenseCost<float>(true); }
TEST(DenseCostTest, quadraticPlusLinearDouble) { checkDenseCost<double>(true); }
