#include <gtest/gtest.h>
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {

	protected:
	    std::unique_ptr<ScenarioTree> m_mockTree;
        size_t m_numNonleafNodes;
        size_t m_numNodes;
        size_t m_numStages;
        std::vector<size_t> m_hostDataIntNumNonleafNodes;
        std::vector<size_t> m_hostDataIntNumNodes;
        std::vector<real_t> m_hostDataRealNumNodes;
        std::vector<size_t> m_hostDataIntNumStages;

	    MarkovTreeTest() {
	        std::ifstream tree_data("../default_tree_data.json");
	        auto treeTemp = std::make_unique<ScenarioTree>(tree_data);
	        m_mockTree = std::move(treeTemp);
            m_numNonleafNodes = m_mockTree->numNonleafNodes();
            m_numNodes = m_mockTree->numNodes();
            m_numStages = m_mockTree->numStages();
            m_hostDataIntNumNonleafNodes.resize(m_numNonleafNodes);
            m_hostDataIntNumNodes.resize(m_numNodes);
            m_hostDataRealNumNodes.resize(m_numNodes);
            m_hostDataIntNumStages.resize(m_numStages);
        };

        virtual ~MarkovTreeTest() {}
};


TEST_F(MarkovTreeTest, Type) {
    EXPECT_TRUE(m_mockTree->isMarkovian());
    EXPECT_FALSE(m_mockTree->isIid());
}

TEST_F(MarkovTreeTest, Sizes) {
    EXPECT_EQ(m_mockTree->numEvents(), 2);
    EXPECT_EQ(m_mockTree->numNonleafNodes(), 3);
    EXPECT_EQ(m_mockTree->numNodes(), 7);
    EXPECT_EQ(m_mockTree->numStages(), 3);
}

TEST_F(MarkovTreeTest, GetStage) {
    m_mockTree->stages().download(m_hostDataIntNumNodes);
    EXPECT_EQ(m_hostDataIntNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetAncestor) {
    m_mockTree->ancestors().download(m_hostDataIntNumNodes);
    EXPECT_EQ(m_hostDataIntNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetProbability) {
    m_mockTree->probabilities().download(m_hostDataRealNumNodes);
    EXPECT_FLOAT_EQ(m_hostDataRealNumNodes[4], 0.4);
}

TEST_F(MarkovTreeTest, GetCondProb) {
    m_mockTree->conditionalProbabilities().download(m_hostDataRealNumNodes);
    EXPECT_FLOAT_EQ(m_hostDataRealNumNodes[5], 0.5);
}

TEST_F(MarkovTreeTest, GetEvent) {
    m_mockTree->events().download(m_hostDataIntNumNodes);
    EXPECT_EQ(m_hostDataIntNumNodes[4], 1);
}

TEST_F(MarkovTreeTest, GetChildFrom) {
    m_mockTree->childFrom().download(m_hostDataIntNumNonleafNodes);
    EXPECT_EQ(m_hostDataIntNumNonleafNodes[2], 5);
}

TEST_F(MarkovTreeTest, GetChildTo) {
    m_mockTree->childTo().download(m_hostDataIntNumNonleafNodes);
    EXPECT_EQ(m_hostDataIntNumNonleafNodes[2], 6);
}

TEST_F(MarkovTreeTest, GetStageFrom) {
    m_mockTree->stageFrom().download(m_hostDataIntNumStages);
    EXPECT_EQ(m_hostDataIntNumStages[2], 3);
}

TEST_F(MarkovTreeTest, GetStageTo) {
    m_mockTree->stageTo().download(m_hostDataIntNumStages);
    EXPECT_EQ(m_hostDataIntNumStages[2], 6);
}
