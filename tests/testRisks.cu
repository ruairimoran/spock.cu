#include <gtest/gtest.h>
#include "../src/risks.cuh"
#include "../src/tree.cuh"


class RisksTest : public testing::Test {

protected:
    std::unique_ptr<ScenarioTree<DEFAULT_FPX>> m_tree;

    /** Prepare some host and device data */
    size_t m_node = 2;
    size_t m_nProj = 0;
    size_t m_nB = 0;
    std::unique_ptr<DTensor<DEFAULT_FPX>> m_d_proj = nullptr;
    std::unique_ptr<DTensor<DEFAULT_FPX>> m_d_b = nullptr;

    RisksTest() {
        std::ifstream tree_data("../../tests/testTreeData.json");
        m_tree = std::make_unique<ScenarioTree<DEFAULT_FPX>>(tree_data);
        m_nProj = m_tree->numChildren()[m_node] * 2 + 1;
        m_nB = m_tree->numChildren()[m_node] * 2 + 1;
        DTensor<DEFAULT_FPX> proj = DTensor<DEFAULT_FPX>::createRandomTensor(m_nProj, m_nProj, 1, -10, 10);
        m_d_proj = std::make_unique<DTensor<DEFAULT_FPX>>(proj);
        DTensor<DEFAULT_FPX> b = DTensor<DEFAULT_FPX>::createRandomTensor(m_nB, 1, 1, -10, 10);
        m_d_b = std::make_unique<DTensor<DEFAULT_FPX>>(b);
    }

    virtual ~RisksTest() {}
};

TEST_F(RisksTest, AvarConeProject) {
    size_t node = 2;
    AVaR myRisk(node, m_tree->numChildren()[node], *m_d_proj, *m_d_b);
}
