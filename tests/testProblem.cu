#include <gtest/gtest.h>
#include "../src/problem.cuh"
#include <fstream>


class ProblemTest : public testing::Test {
protected:
    ProblemTest() = default;
};


TEMPLATE_WITH_TYPE_T
class ProblemTestData {

public:
    std::string m_path = "../../data/";
    std::unique_ptr<ScenarioTree<T>> m_tree;
    std::unique_ptr<ProblemData<T>> m_data;

    ProblemTestData() {
        m_tree = std::make_unique<ScenarioTree<T>>(m_path);
        m_data = std::make_unique<ProblemData<T>>(*m_tree);
    };

    virtual ~ProblemTestData() = default;
};


TEMPLATE_WITH_TYPE_T
void testSizes(ProblemTestData<T> &d) {
    EXPECT_EQ(d.m_data->numStates(), 3);
    EXPECT_EQ(d.m_data->numInputs(), 2);
}

TEST_F(ProblemTest, sizes) {
    ProblemTestData<float> df;
    testSizes<float>(df);
    ProblemTestData<double> dd;
    testSizes<double>(dd);
}