#include <gtest/gtest.h>
#include "../src/problem.cuh"
#include <fstream>


class ProblemDataTest : public testing::Test {

protected:
    std::unique_ptr<ScenarioTree<DEFAULT_FPX>> m_tree;
    std::unique_ptr<ProblemData<DEFAULT_FPX>> m_data;

    ProblemDataTest() {
        m_tree = std::make_unique<ScenarioTree<DEFAULT_FPX>>("../../data/");
        m_data = std::make_unique<ProblemData<DEFAULT_FPX>>(*m_tree);
    };

    virtual ~ProblemDataTest() {}
};


TEST_F(ProblemDataTest, Sizes) {
    EXPECT_EQ(m_data->numStates(), 3);
    EXPECT_EQ(m_data->numInputs(), 2);
}