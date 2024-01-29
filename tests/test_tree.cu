#include <gtest/gtest.h>
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {
	protected:
	    std::unique_ptr<ScenarioTree> m_mockMarkovTree;

	    MarkovTreeTest(){
	        std::ifstream m_tree_data("../src/tree_data.json");
	        auto m_mockMarkovTreeTemp = std::make_unique<ScenarioTree>(m_tree_data);
            m_mockMarkovTree = std::move(m_mockMarkovTreeTemp);
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
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_stage_of_node(-1));
//     EXPECT_EQ(s_mockMarkovTree.get_stage_of_node(5), 2);
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_stage_of_node(s_mockMarkovTree.num_nodes()));
// }

TEST_F(MarkovTreeTest, GetAncestorOfNode) {
    // EXPECT_ANY_THROW(s_mockMarkovTree.getAncestorOfNode(0));
    EXPECT_EQ(m_mockMarkovTree->ancestors()[5], 2);
    // EXPECT_ANY_THROW(s_mockMarkovTree.getAncestorOfNode(s_mockMarkovTree.num_nodes()));
}

// TEST_F(MarkovTreeTest, GetProbability) {
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_probability_of_node(-1));
//     EXPECT_FLOAT_EQ(s_mockMarkovTree.get_probability_of_node(5), 0.2);
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_probability_of_node(s_mockMarkovTree.num_nodes()));
// }

// TEST_F(MarkovTreeTest, GetEvent) {
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_event_of_node(0));
//     EXPECT_EQ(s_mockMarkovTree.get_event_of_node(4), 1);
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_event_of_node(s_mockMarkovTree.num_nodes()));
// }

// TEST_F(MarkovTreeTest, GetChildrenOf) {
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_children_of_node(-1));
//     EXPECT_TRUE((s_mockMarkovTree.get_children_of_node(2) == std::vector<int> {5, 6}));
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_children_of_node(s_mockMarkovTree.num_nonleaf_nodes()));
// }

// TEST_F(MarkovTreeTest, GetCondProbOfChildren) {
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_cond_prob_of_children_of_node(-1));
//     EXPECT_TRUE((s_mockMarkovTree.get_cond_prob_of_children_of_node(1) == std::vector<double> {0.5, 0.5}));
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_cond_prob_of_children_of_node(s_mockMarkovTree.num_nonleaf_nodes()));
// }

// TEST_F(MarkovTreeTest, GetNodesAtStage) {
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_nodes_of_stage(-1));
//     EXPECT_TRUE((s_mockMarkovTree.get_nodes_of_stage(2) == std::vector<int> {3, 4, 5, 6}));
//     EXPECT_ANY_THROW(s_mockMarkovTree.get_nodes_of_stage(s_mockMarkovTree.num_nodes()));
// }
