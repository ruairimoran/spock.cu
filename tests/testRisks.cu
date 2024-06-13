#include <gtest/gtest.h>
#include "../src/risks.cuh"
#include "../src/tree.cuh"


class RisksTest : public testing::Test {

protected:
    std::unique_ptr<ScenarioTree<DEFAULT_FPX>> m_tree;

    /** Prepare some host and device data */
    size_t m_node = 2;
    size_t m_n = 0;
    std::unique_ptr<DTensor<DEFAULT_FPX>> m_d_data = nullptr;
    std::vector<DEFAULT_FPX> m_hostData;
    std::vector<DEFAULT_FPX> m_hostTest;

    RisksTest() {
        std::ifstream tree_data("../../tests/testTreeData.json");
        m_tree = std::make_unique<ScenarioTree<DEFAULT_FPX>>(tree_data);
        m_n = m_tree->numChildren()[m_node] * 2 + 1;
        m_d_data = std::make_unique<DTensor<DEFAULT_FPX>>(m_n);
        m_hostData.resize(m_n);
        m_hostTest.resize(m_n);
        /** Positive and negative values in m_data */
        for (size_t i = 0; i < m_n; i = i + 2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i = 1; i < m_n; i = i + 2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data->upload(m_hostData);
    }

    virtual ~RisksTest() {}
};

TEST_F(RisksTest, AvarConeProject) {
    size_t node = 2;
    AVaR myRisk(node, m_tree->numChildren()[node], *m_d_data);
}
