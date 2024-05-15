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
