#include <gtest/gtest.h>
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {

	protected:
	    std::unique_ptr<ScenarioTree> mockTree;
        bool isMarkovian;
        bool isIid;
        int numNonleafNodes;
        int numNodes;
        int numStages;
        std::vector<int> hostDataIntNumNonleafNodes;
        std::vector<int> hostDataIntNumNodes;
        std::vector<real_t> hostDataRealNumNodes;
        std::vector<int> hostDataIntNumStages;

	    MarkovTreeTest() {
	        std::ifstream tree_data("../tree_data.json");
	        auto treeTemp = std::make_unique<ScenarioTree>(tree_data);
	        mockTree = std::move(treeTemp);
            numNonleafNodes = mockTree->numNonleafNodes();
            numNodes = mockTree->numNodes();
            numStages = mockTree->numStages();
            hostDataIntNumNonleafNodes.resize(numNonleafNodes);
            hostDataIntNumNodes.resize(numNodes);
            hostDataRealNumNodes.resize(numNodes);
            hostDataIntNumStages.resize(numStages);
        };

        virtual ~MarkovTreeTest() {}
};


TEST_F(MarkovTreeTest, Type) {
    EXPECT_TRUE(mockTree->isMarkovian());
    EXPECT_FALSE(mockTree->isIid());
}

TEST_F(MarkovTreeTest, Sizes) {
    EXPECT_EQ(mockTree->numNonleafNodes(), 3);
    EXPECT_EQ(mockTree->numNodes(), 7);
    EXPECT_EQ(mockTree->numStages(), 3);
}

TEST_F(MarkovTreeTest, GetStage) {
    mockTree->stages().download(hostDataIntNumNodes);
    EXPECT_EQ(hostDataIntNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetAncestor) {
    mockTree->ancestors().download(hostDataIntNumNodes);
    EXPECT_EQ(hostDataIntNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetProbability) {
    mockTree->probabilities().download(hostDataRealNumNodes);
    EXPECT_FLOAT_EQ(hostDataRealNumNodes[4], 0.4);
}

TEST_F(MarkovTreeTest, GetCondProb) {
    mockTree->conditionalProbabilities().download(hostDataRealNumNodes);
    EXPECT_FLOAT_EQ(hostDataRealNumNodes[5], 0.5);
}

TEST_F(MarkovTreeTest, GetEvent) {
    mockTree->events().download(hostDataIntNumNodes);
    EXPECT_EQ(hostDataIntNumNodes[4], 1);
}

TEST_F(MarkovTreeTest, GetChildFrom) {
    mockTree->childFrom().download(hostDataIntNumNonleafNodes);
    EXPECT_EQ(hostDataIntNumNonleafNodes[2], 5);
}

TEST_F(MarkovTreeTest, GetChildTo) {
    mockTree->childTo().download(hostDataIntNumNonleafNodes);
    EXPECT_EQ(hostDataIntNumNonleafNodes[2], 6);
}

TEST_F(MarkovTreeTest, GetStageFrom) {
    mockTree->stageFrom().download(hostDataIntNumStages);
    EXPECT_EQ(hostDataIntNumStages[2], 3);
}

TEST_F(MarkovTreeTest, GetStageTo) {
    mockTree->stageTo().download(hostDataIntNumStages);
    EXPECT_EQ(hostDataIntNumStages[2], 6);
}
