#include <gtest/gtest.h>
#include "../include/stdgpu.h"
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {
	protected:
	    std::unique_ptr<ScenarioTree> m_mockTree;
        bool isMarkovian;
        bool isIid;
        int numNonleafNodes;
        int numNodes;
        int numStages;
        int* hostDataIntNumNonleafNodes;
        int* hostDataIntNumNodes;
        real_t* hostDataRealNumNodes;
        int* hostDataIntNumStages;

	    MarkovTreeTest(){
	        std::ifstream tree_data("../../src/tree_data.json");
	        auto treeTemp = std::make_unique<ScenarioTree>(tree_data);
	        m_mockTree = std::move(treeTemp);
            numNonleafNodes = m_mockTree->numNonleafNodes();
            numNodes = m_mockTree->numNodes();
            numStages = m_mockTree->numStages();
            hostDataIntNumNonleafNodes = new int[numNonleafNodes];
            hostDataIntNumNodes = new int[numNodes];
            hostDataRealNumNodes = new real_t[numNodes];
            hostDataIntNumStages = new int[numStages];
        };

        virtual ~MarkovTreeTest() {
        }
};


TEST_F(MarkovTreeTest, Type) {
    EXPECT_TRUE(m_mockTree->isMarkovian());
    EXPECT_FALSE(m_mockTree->isIid());
}

TEST_F(MarkovTreeTest, Sizes) {
    EXPECT_EQ(m_mockTree->numNonleafNodes(), 3);
    EXPECT_EQ(m_mockTree->numNodes(), 7);
    EXPECT_EQ(m_mockTree->numStages(), 3);
}

TEST_F(MarkovTreeTest, GetStage) {
    cudaMemcpy(hostDataIntNumNodes, m_mockTree->stages(), numNodes*sizeof(int), D2H);
    EXPECT_EQ(hostDataIntNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetAncestor) {
    cudaMemcpy(hostDataIntNumNodes, m_mockTree->ancestors(), numNodes*sizeof(int), D2H);
    EXPECT_EQ(hostDataIntNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, GetProbability) {
    cudaMemcpy(hostDataRealNumNodes, m_mockTree->probabilities(), numNodes*sizeof(real_t), D2H);
    EXPECT_FLOAT_EQ(hostDataRealNumNodes[4], 0.4);
}

TEST_F(MarkovTreeTest, GetEvent) {
    cudaMemcpy(hostDataIntNumNodes, m_mockTree->events(), numNodes*sizeof(int), D2H);
    EXPECT_EQ(hostDataIntNumNodes[4], 1);
}

TEST_F(MarkovTreeTest, GetChildFrom) {
    cudaMemcpy(hostDataIntNumNonleafNodes, m_mockTree->childFrom(), numNonleafNodes*sizeof(int), D2H);
    EXPECT_EQ(hostDataIntNumNonleafNodes[2], 5);
}

TEST_F(MarkovTreeTest, GetChildTo) {
    cudaMemcpy(hostDataIntNumNonleafNodes, m_mockTree->childTo(), numNonleafNodes*sizeof(int), D2H);
    EXPECT_EQ(hostDataIntNumNonleafNodes[2], 6);
}

//TEST_F(MarkovTreeTest, GetCondProb) {
//    cudaMemcpy(hostDataRealNumNodes, m_mockTree->conditionalProbabilities(), numNodes*sizeof(real_t), D2H);
//    EXPECT_EQ(hostDataRealNumNodes[0], 0.5);
//}
//
//TEST_F(MarkovTreeTest, GetStageFrom) {
//    cudaMemcpy(hostDataIntNumStages, m_mockTree->stageFrom(), numStages*sizeof(int), D2H);
//    EXPECT_EQ(hostDataIntNumStages[2], 3);
//}
//
//TEST_F(MarkovTreeTest, GetStageTo) {
//    cudaMemcpy(hostDataIntNumStages, m_mockTree->stageTo(), numStages*sizeof(int), D2H);
//    EXPECT_EQ(hostDataIntNumStages[2], 6);
//}
