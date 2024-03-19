#include <gtest/gtest.h>
#include "../src/risks.cuh"
#include "../src/tree.cuh"


class RisksTest : public testing::Test {

protected:
    Context m_context;  ///< Create one context only
    std::unique_ptr<ScenarioTree> m_tree;

    /** Prepare some host and device data */
    size_t m_node = 2;
    size_t m_n = 0;
    DeviceVector<real_t> m_d_data;
    std::vector<real_t> m_hostData;
    std::vector<real_t> m_hostTest;
    RisksTest() {
        std::ifstream tree_data("../../tests/default_tree_data.json");
        m_tree = std::make_unique<ScenarioTree>(tree_data);
        m_n = m_tree->numChildren().fetchElementFromDevice(m_node) * 2 + 1;
        m_d_data.allocateOnDevice(m_n);
        m_hostData.resize(m_n);
        m_hostTest.resize(m_n);
        /** Positive and negative values in m_hostData */
        for (size_t i=0; i<m_n; i=i+2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i=1; i<m_n; i=i+2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);
    }

    virtual ~RisksTest() {}
};


TEST_F(RisksTest, AvarConeProject) {
    AVaR myRisk(m_context,
                2,
                0.98,
                m_tree->numChildren(),
                m_tree->childFrom(),
                m_tree->conditionalProbabilities());
    myRisk.cone().project(m_d_data);
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest != m_hostData));
}
