#include <gtest/gtest.h>
#include "problem.cuh"
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

/* ---------------------------------------
 * Project on dynamics (online)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testDynamicsProjectionOnline(ProblemTestData<T> &d, T epsilon) {
    size_t statesSize = d.m_tree->numStates() * d.m_tree->numNodes();
    size_t inputsSize = d.m_tree->numInputs() * d.m_tree->numNonleafNodes();
    std::vector<T> cvxStates(statesSize);
    std::vector<T> cvxInputs(inputsSize);
    std::string ext = d.m_tree->fpFileExt();
    DTensor<T> dpStates = DTensor<T>::parseFromFile(d.m_path + "dpTestStates" + ext);
    DTensor<T> dpInputs = DTensor<T>::parseFromFile(d.m_path + "dpTestInputs" + ext);
    DTensor<T> dpProjectedStates = DTensor<T>::parseFromFile(d.m_path + "dpProjectedStates" + ext);
    DTensor<T> dpProjectedInputs = DTensor<T>::parseFromFile(d.m_path + "dpProjectedInputs" + ext);
    dpProjectedStates.download(cvxStates);
    dpProjectedInputs.download(cvxInputs);
    DTensor<T> d_initState(dpStates, 0, 0, d.m_tree->numStates() - 1);
    DTensor<T> d_states(d.m_tree->numStates(), 1, d.m_tree->numNodes());
    DTensor<T> d_inputs(d.m_tree->numInputs(), 1, d.m_tree->numNonleafNodes());
    dpStates.deviceCopyTo(d_states);
    dpInputs.deviceCopyTo(d_inputs);
    d.m_data->dynamics()->project(d_initState, d_states, d_inputs);
    /* Compare states */
    std::vector<T> spockStates(statesSize);
    d_states.download(spockStates);
    for (size_t i = 0; i < statesSize; i++) { EXPECT_NEAR(spockStates[i], cvxStates[i], epsilon); }
    /* Compare inputs */
    std::vector<T> spockInputs(inputsSize);
    d_inputs.download(spockInputs);
    for (size_t i = 0; i < inputsSize; i++) { EXPECT_NEAR(spockInputs[i], cvxInputs[i], epsilon); }
}

TEST_F(ProblemTest, dynamicsProjectionOnline) {
    ProblemTestData<float> df;
    testDynamicsProjectionOnline<float>(df, TEST_PRECISION_LOW);
    ProblemTestData<double> dd;
    testDynamicsProjectionOnline<double>(dd, TEST_PRECISION_HIGH);
}