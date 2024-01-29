#include <gtest/gtest.h>
#include "../src/tree.cuh"


// ***********************************************************************************
// ---- IF ANY methods suggested in develop PR are used, I get `Exception: SegFault`
//************************************************************************************


TEST(MarkovTreeTest, All) {
    std::ifstream treeData("../../src/tree_data.json");
	ScenarioTree mockTree(treeData);

    // EXPECT_TRUE(mockTree.is_markovian());
    // EXPECT_FALSE(mockTree.is_iid())
    // EXPECT_EQ(mockTree.num_nonleaf_nodes(), 3);
    // EXPECT_EQ(mockTree.num_nodes(), 7);
    // EXPECT_EQ(mockTree.num_stages(), 3);
    // EXPECT_EQ(mockTree.get_stage_of_node(5), 2);
    EXPECT_EQ(mockTree.ancestors()[5], 2);
    // EXPECT_FLOAT_EQ(mockTree.get_probability_of_node(5), 0.2);
    // EXPECT_EQ(mockTree.get_event_of_node(4), 1);
    // EXPECT_TRUE((mockTree.get_children_of_node(2) == std::vector<int> {5, 6}));
    // EXPECT_TRUE((mockTree.get_cond_prob_of_children_of_node(1) == std::vector<double> {0.5, 0.5}));
    // EXPECT_TRUE((mockTree.get_nodes_of_stage(2) == std::vector<int> {3, 4, 5, 6}));
}


// class MarkovTreeTest : public testing::Test {
// 	protected:
// 	    std::ifstream tree_data{"../../src/tree_data.json"};
// 	    ScenarioTree m_mockTree{tree_data};
// };


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

// TEST_F(MarkovTreeTest, GetAncestorOfNode) {
//     EXPECT_EQ(m_mockTree.ancestors()[5], 2);
// }

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
