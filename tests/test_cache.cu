#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <fstream>


class CacheTest : public testing::Test {

protected:
    Context m_context;  ///< Create one context only
    std::unique_ptr<ScenarioTree> m_tree;
    std::unique_ptr<ProblemData> m_data;
    std::unique_ptr<Cache> m_cache;

    /** Prepare some host and device data */
    size_t m_n = 64;
    real_t m_tol = 1e-4;
    size_t m_maxIters = 20;
    DeviceVector<real_t> m_d_data = DeviceVector<real_t>(m_context, m_n);;
    std::vector<real_t> m_hostData = std::vector<real_t>(m_n);;
    std::vector<real_t> m_hostTest = std::vector<real_t>(m_n);
    CacheTest() {
        std::ifstream tree_data("../../tests/default_tree_data.json");
        m_tree = std::make_unique<ScenarioTree>(m_context, tree_data);
        std::ifstream problem_data("../../tests/default_problem_data.json");
        m_data = std::make_unique<ProblemData>(m_context, *m_tree, problem_data);
        m_cache = std::make_unique<Cache>(m_context, *m_tree, *m_data, m_tol, m_maxIters);

        /** Positive and negative values in m_hostData */
        for (size_t i=0; i<m_n; i=i+2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i=1; i<m_n; i=i+2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);
    };

    virtual ~CacheTest() {}
};


TEST_F(CacheTest, Uno) {
    EXPECT_EQ(1, true);
}
