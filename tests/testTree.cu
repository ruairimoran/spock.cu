#include <gtest/gtest.h>
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {

protected:
    std::unique_ptr<ScenarioTree<DEFAULT_FPX>> m_mockTree;
    size_t m_numNonleafNodes;
    size_t m_numNodes;
    size_t m_numStages;
    std::vector<size_t> m_intNumNonleafNodes;
    std::vector<size_t> m_intNumNodes;
    std::vector<DEFAULT_FPX> m_realNumNodes;
    std::vector<size_t> m_intNumStages;

    MarkovTreeTest() {
        std::ifstream tree_data("../../tests/testTreeData.json");
        m_mockTree = std::make_unique<ScenarioTree<DEFAULT_FPX>>(tree_data);
        m_numNonleafNodes = m_mockTree->numNonleafNodes();
        m_numNodes = m_mockTree->numNodes();
        m_numStages = m_mockTree->numStages();
        m_intNumNonleafNodes.resize(m_numNonleafNodes);
        m_intNumNodes.resize(m_numNodes);
        m_realNumNodes.resize(m_numNodes);
        m_intNumStages.resize(m_numStages);
    };

    virtual ~MarkovTreeTest() {}
};

TEST_F(MarkovTreeTest, Sizes) {
    EXPECT_EQ(m_mockTree->numEvents(), 2);
    EXPECT_EQ(m_mockTree->numNonleafNodes(), 3);
    EXPECT_EQ(m_mockTree->numNodes(), 7);
    EXPECT_EQ(m_mockTree->numStages(), 3);
}

TEST_F(MarkovTreeTest, GetStage) {
    m_mockTree->d_stages().download(m_intNumNodes);
    EXPECT_EQ(m_intNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetAncestor) {
    m_mockTree->d_ancestors().download(m_intNumNodes);
    EXPECT_EQ(m_intNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetProbability) {
    m_mockTree->d_probabilities().download(m_realNumNodes);
    EXPECT_FLOAT_EQ(m_realNumNodes[4], 0.4);
}

TEST_F(MarkovTreeTest, GetCondProb) {
    m_mockTree->d_conditionalProbabilities().download(m_realNumNodes);
    EXPECT_FLOAT_EQ(m_realNumNodes[5], 0.5);
}

TEST_F(MarkovTreeTest, GetEvent) {
    m_mockTree->d_events().download(m_intNumNodes);
    EXPECT_EQ(m_intNumNodes[4], 1);
}

TEST_F(MarkovTreeTest, GetChildFrom) {
    m_mockTree->d_childFrom().download(m_intNumNonleafNodes);
    EXPECT_EQ(m_intNumNonleafNodes[2], 5);
}

TEST_F(MarkovTreeTest, GetChildTo) {
    m_mockTree->d_childTo().download(m_intNumNonleafNodes);
    EXPECT_EQ(m_intNumNonleafNodes[2], 6);
}

TEST_F(MarkovTreeTest, GetNumChildren) {
    m_mockTree->d_numChildren().download(m_intNumNonleafNodes);
    EXPECT_EQ(m_intNumNonleafNodes[2], 2);
}

TEST_F(MarkovTreeTest, GetStageFrom) {
    m_mockTree->d_nodeFrom().download(m_intNumStages);
    EXPECT_EQ(m_intNumStages[2], 3);
}

TEST_F(MarkovTreeTest, GetStageTo) {
    m_mockTree->d_nodeTo().download(m_intNumStages);
    EXPECT_EQ(m_intNumStages[2], 6);
}
