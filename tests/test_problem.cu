#include <gtest/gtest.h>
#include "../src/problem.cuh"
#include <fstream>


class ProblemDataTest : public testing::Test {

protected:
    std::unique_ptr<ScenarioTree> m_tree;
    std::unique_ptr<ProblemData> m_data;

    ProblemDataTest() {
        std::ifstream tree_data("../../tests/default_tree_data.json");
        m_tree = std::make_unique<ScenarioTree>(tree_data);
        std::ifstream problem_data("../../tests/default_problem_data.json");
        m_data = std::make_unique<ProblemData>(*m_tree, problem_data);
    };

    virtual ~ProblemDataTest() {}
};


TEST_F(ProblemDataTest, Sizes) {
    EXPECT_EQ(m_data->numStates(), 3);
    EXPECT_EQ(m_data->numInputs(), 2);
}