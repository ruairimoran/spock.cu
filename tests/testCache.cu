#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <fstream>


class CacheTest : public testing::Test {

protected:
    std::unique_ptr<ScenarioTree<DEFAULT_FPX>> m_tree;
    std::unique_ptr<ProblemData<DEFAULT_FPX>> m_data;
    std::unique_ptr<Cache<DEFAULT_FPX>> m_cache;

    /** Prepare some host and device data */
    size_t m_n = 64;
    DEFAULT_FPX m_tol = 1e-4;
    size_t m_maxIters = 20;
    DTensor<DEFAULT_FPX> m_d_data = DTensor<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostData = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostTest = std::vector<DEFAULT_FPX>(m_n);

    CacheTest() {
        std::ifstream tree_data("../../tests/testTreeData.json");
        m_tree = std::make_unique<ScenarioTree<DEFAULT_FPX>>(tree_data);
        std::ifstream problem_data("../../tests/testProblemData.json");
        m_data = std::make_unique<ProblemData<DEFAULT_FPX>>(*m_tree, problem_data);
        m_cache = std::make_unique<Cache<DEFAULT_FPX>>(*m_tree, *m_data, m_tol, m_maxIters);

        /** Positive and negative values in m_data */
        for (size_t i = 0; i < m_n; i = i + 2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i = 1; i < m_n; i = i + 2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);
    };

    virtual ~CacheTest() {}
};

TEST_F(CacheTest, InitialiseState) {
    std::vector<DEFAULT_FPX> initialState = {3., 5., 4.};
    m_cache->initialiseState(initialState);
    std::vector<DEFAULT_FPX> sol(m_cache->solutionSize());
    m_cache->solution().download(sol);
    EXPECT_EQ(sol, initialState);
}

TEST_F(CacheTest, DynamicsProjectionOnline) {
    std::vector<DEFAULT_FPX> initialState = {3., 5., 4.};
    m_cache->initialiseState(initialState);
    m_cache->projectOnDynamics();
    size_t xuSize = m_tree->numNodes() * m_data->numStates() + m_tree->numNonleafNodes() * m_data->numInputs();
    DTensor<DEFAULT_FPX> xu(m_cache->solution(), 0, 0, xuSize - 1);
    std::vector<DEFAULT_FPX> sol(xuSize);
    xu.download(sol);

//    prim = mock_cache.get_primal();  // template
//    for i in range(seg_p[1], seg_p[3]):
//    prim[i] = np.random.randn(prim[i].size).reshape(-1, 1)
//
//    /* Solve with dp */
//    mock_cache.cache_initial_state(prim[seg_p[1]])
//    mock_cache.set_primal(prim)
//    mock_cache.project_on_dynamics()
//    dp_result = mock_cache.get_primal()
//    x_dp = np.asarray(dp_result[seg_p[1]: seg_p[2]])[:, :, 0]
//    u_dp = np.asarray(dp_result[seg_p[2]: seg_p[3]])[:, :, 0]
//    /* ensure x0 stayed the same */
//    self.assertTrue(np.allclose(prim[seg_p[1]].T, x_dp[0]))
//
//    /* Solve with cvxpy */
//    for i in range(seg_p[1], seg_p[3]):
//    prim[i] = prim[i].reshape(-1,)
//
//    x_bar = np.asarray(prim[seg_p[1]: seg_p[2]])
//    u_bar = np.asarray(prim[seg_p[2]: seg_p[3]])
//    N = self.__tree_from_markov.num_nodes
//    n = self.__tree_from_markov.num_nonleaf_nodes
//    x = cp.Variable(x_bar.shape)
//    u = cp.Variable(u_bar.shape)
//    /* Sum problem objectives and concatenate constraints */
//    cost = 0
//    constraints = [x[0] == x_bar[0]]
//    /* Nonleaf nodes */
//    for node in range(n):
//    cost += cp.sum_squares(x[node] - x_bar[node]) + cp.sum_squares(u[node] - u_bar[node])
//    for ch in self.__tree_from_markov.children_of(node):
//    constraints += [x[ch] ==
//                    self.__raocp_from_markov.state_dynamics_at_node(ch) @ x[node] +
//                                                                          self.__raocp_from_markov.control_dynamics_at_node(ch) @ u[node]]
//
//    /* Leaf nodes */
//    for node in range(n, N):
//    cost += cp.sum_squares(x[node] - x_bar[node])
//
//    problem = cp.Problem(cp.Minimize(cost), constraints)
//    problem.solve()
//    /* Ensure x0 stayed the same */
//    self.assertTrue(np.allclose(prim[seg_p[1]], x.value[0]))
//
//    /* Check solutions are similar */
//    self.assertTrue(np.allclose(x.value, x_dp))
//    self.assertTrue(np.allclose(u.value, u_dp))
}
