#include <gtest/gtest.h>
#include "../src/tree.h"


class MarkovTreeTest : public testing::Test {
    protected:
        ScenarioTree mock_markov_tree = ScenarioTree();
};

TEST_F(MarkovTreeTest, Type) {
    EXPECT_TRUE(mock_markov_tree.is_markovian());
    EXPECT_FALSE(mock_markov_tree.is_iid());
}

TEST_F(MarkovTreeTest, Sizes) {
    EXPECT_EQ(mock_markov_tree.num_nonleaf_nodes(), 3);
    EXPECT_EQ(mock_markov_tree.num_nodes(), 7);
    EXPECT_EQ(mock_markov_tree.num_stages(), 3);
}

TEST_F(MarkovTreeTest, GetStage) {
    EXPECT_ANY_THROW(mock_markov_tree.get_stage_of_node(-1));
    EXPECT_EQ(mock_markov_tree.get_stage_of_node(5), 2);
    EXPECT_ANY_THROW(mock_markov_tree.get_stage_of_node(mock_markov_tree.num_nodes()));
}

TEST_F(MarkovTreeTest, GetAncestor) {
    EXPECT_ANY_THROW(mock_markov_tree.get_ancestor_of_node(0));
    EXPECT_EQ(mock_markov_tree.get_ancestor_of_node(5), 2);
    EXPECT_ANY_THROW(mock_markov_tree.get_ancestor_of_node(mock_markov_tree.num_nodes()));
}

TEST_F(MarkovTreeTest, GetProbability) {
    EXPECT_ANY_THROW(mock_markov_tree.get_probability_of_node(-1));
    EXPECT_FLOAT_EQ(mock_markov_tree.get_probability_of_node(5), 0.2);
    EXPECT_ANY_THROW(mock_markov_tree.get_probability_of_node(mock_markov_tree.num_nodes()));
}

TEST_F(MarkovTreeTest, GetEvent) {
    EXPECT_ANY_THROW(mock_markov_tree.get_event_of_node(0));
    EXPECT_EQ(mock_markov_tree.get_event_of_node(4), 1);
    EXPECT_ANY_THROW(mock_markov_tree.get_event_of_node(mock_markov_tree.num_nodes()));
}

TEST_F(MarkovTreeTest, GetChildrenOf) {
    EXPECT_ANY_THROW(mock_markov_tree.get_children_of_node(-1));
    EXPECT_TRUE((mock_markov_tree.get_children_of_node(2) == std::vector<int> {5, 6}));
    EXPECT_ANY_THROW(mock_markov_tree.get_children_of_node(mock_markov_tree.num_nonleaf_nodes()));
}

TEST_F(MarkovTreeTest, GetCondProbOfChildren) {
    EXPECT_ANY_THROW(mock_markov_tree.get_cond_prob_of_children_of_node(-1));
    EXPECT_TRUE((mock_markov_tree.get_cond_prob_of_children_of_node(1) == std::vector<double> {0.5, 0.5}));
    EXPECT_ANY_THROW(mock_markov_tree.get_cond_prob_of_children_of_node(mock_markov_tree.num_nonleaf_nodes()));
}

TEST_F(MarkovTreeTest, GetNodesAtStage) {
    EXPECT_ANY_THROW(mock_markov_tree.get_nodes_of_stage(-1));
    EXPECT_TRUE((mock_markov_tree.get_nodes_of_stage(2) == std::vector<int> {3, 4, 5, 6}));
    EXPECT_ANY_THROW(mock_markov_tree.get_nodes_of_stage(mock_markov_tree.num_nodes()));
}
