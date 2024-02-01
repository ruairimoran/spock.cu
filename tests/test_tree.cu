#include <gtest/gtest.h>
#include "../include/stdgpu.h"
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {
	protected:
	    std::unique_ptr<ScenarioTree> m_mockTree;
        int numNodes;
        int* hostDataSizeNumNodes;
        int* hostDataSizeNumNonleafNodes;

	    MarkovTreeTest(){
	        std::ifstream tree_data("../../src/tree_data.json");
	        auto treeTemp = std::make_unique<ScenarioTree>(tree_data);
	        m_mockTree = std::move(treeTemp);
            numNodes = m_mockTree->numNodes();
            hostDataSizeNumNodes = new int[numNodes];
            hostDataSizeNumNonleafNodes = new int[numNodes];
        };

        virtual ~MarkovTreeTest() {
        }
};


// TEST_F(MarkovTreeTest, Type) {
//     EXPECT_TRUE(s_mockMarkovTree.is_markovian());
//     EXPECT_FALSE(s_mockMarkovTree.is_iid());
// }

// TEST_F(MarkovTreeTest, Sizes) {
//     EXPECT_EQ(s_mockMarkovTree.num_nonleaf_nodes(), 3);
//     EXPECT_EQ(s_mockMarkovTree.num_nodes(), 7);
//     EXPECT_EQ(s_mockMarkovTree.num_stages(), 3);
// }

// TEST_F(MarkovTreeTest, GetStage) {
//     EXPECT_EQ(s_mockMarkovTree.get_stage_of_node(5), 2);
// }

TEST_F(MarkovTreeTest, GetAncestor) {
    cudaMemcpy(hostDataSizeNumNodes, m_mockTree->ancestors(), numNodes*sizeof(int), D2H);
    EXPECT_EQ(hostDataSizeNumNodes[5], 2);
}

// TEST_F(MarkovTreeTest, GetProbability) {
//     EXPECT_FLOAT_EQ(s_mockMarkovTree.get_probability_of_node(5), 0.2);
// }

// TEST_F(MarkovTreeTest, GetEvent) {
//     EXPECT_EQ(s_mockMarkovTree.get_event_of_node(4), 1);
// }

// TEST_F(MarkovTreeTest, GetChildrenOf) {
//     EXPECT_TRUE((s_mockMarkovTree.get_children_of_node(2) == std::vector<int> {5, 6}));
// }

// TEST_F(MarkovTreeTest, GetCondProbOfChildren) {
//     EXPECT_TRUE((s_mockMarkovTree.get_cond_prob_of_children_of_node(1) == std::vector<double> {0.5, 0.5}));
// }

// TEST_F(MarkovTreeTest, GetNodesAtStage) {
//     EXPECT_TRUE((s_mockMarkovTree.get_nodes_of_stage(2) == std::vector<int> {3, 4, 5, 6}));
// }
