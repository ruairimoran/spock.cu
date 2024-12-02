#include <gtest/gtest.h>
#include "../src/tree.cuh"


class MarkovTreeTest : public testing::Test {
protected:
    MarkovTreeTest() = default;
};


TEMPLATE_WITH_TYPE_T
class MarkovTreeTestData {

public:
    std::string m_path = "../../data/";
    std::unique_ptr<ScenarioTree<T>> m_mockTree;
    size_t m_numNonleafNodes;
    size_t m_numNodes;
    size_t m_numStages;
    std::vector<size_t> m_intNumNonleafNodes;
    std::vector<size_t> m_intNumNodes;
    std::vector<T> m_realNumNodes;
    std::vector<size_t> m_intNumStages;

    MarkovTreeTestData() {
        m_mockTree = std::make_unique<ScenarioTree<T>>(m_path);
        m_numNonleafNodes = m_mockTree->numNonleafNodes();
        m_numNodes = m_mockTree->numNodes();
        m_numStages = m_mockTree->numStages();
        m_intNumNonleafNodes = std::vector<size_t>(m_numNonleafNodes);
        m_intNumNodes = std::vector<size_t>(m_numNodes);
        m_realNumNodes = std::vector<T>(m_numNodes);
        m_intNumStages = std::vector<size_t>(m_numStages);
    };

    virtual ~MarkovTreeTestData() = default;
};

TEMPLATE_WITH_TYPE_T
void testSizes(MarkovTreeTestData<T> &d) {
    EXPECT_EQ(d.m_mockTree->numEvents(), 2);
    EXPECT_EQ(d.m_mockTree->numNonleafNodes(), 7);
    EXPECT_EQ(d.m_mockTree->numNodes(), 11);
    EXPECT_EQ(d.m_mockTree->numStages(), 4);
}

TEST_F(MarkovTreeTest, sizes) {
    MarkovTreeTestData<float> df;
    testSizes<float>(df);
    MarkovTreeTestData<double> dd;
    testSizes<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetStage(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_stages().download(d.m_intNumNodes);
    EXPECT_EQ(d.m_intNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, getStage) {
    MarkovTreeTestData<float> df;
    testGetStage<float>(df);
    MarkovTreeTestData<double> dd;
    testGetStage<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetAncestor(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_ancestors().download(d.m_intNumNodes);
    EXPECT_EQ(d.m_intNumNodes[5], 2);
}

TEST_F(MarkovTreeTest, getAncestor) {
    MarkovTreeTestData<float> df;
    testGetAncestor<float>(df);
    MarkovTreeTestData<double> dd;
    testGetAncestor<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetProbability(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_probabilities().download(d.m_realNumNodes);
    EXPECT_FLOAT_EQ(d.m_realNumNodes[4], 0.4);
}

TEST_F(MarkovTreeTest, getProbability) {
    MarkovTreeTestData<float> df;
    testGetProbability<float>(df);
    MarkovTreeTestData<double> dd;
    testGetProbability<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetCondProb(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_conditionalProbabilities().download(d.m_realNumNodes);
    EXPECT_FLOAT_EQ(d.m_realNumNodes[5], 0.5);
}

TEST_F(MarkovTreeTest, getCondProb) {
    MarkovTreeTestData<float> df;
    testGetCondProb<float>(df);
    MarkovTreeTestData<double> dd;
    testGetCondProb<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetEvent(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_events().download(d.m_intNumNodes);
    EXPECT_EQ(d.m_intNumNodes[4], 1);
}

TEST_F(MarkovTreeTest, getEvent) {
    MarkovTreeTestData<float> df;
    testGetEvent<float>(df);
    MarkovTreeTestData<double> dd;
    testGetEvent<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetChildFrom(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_childFrom().download(d.m_intNumNonleafNodes);
    EXPECT_EQ(d.m_intNumNonleafNodes[2], 5);
}

TEST_F(MarkovTreeTest, getChildFrom) {
    MarkovTreeTestData<float> df;
    testGetChildFrom<float>(df);
    MarkovTreeTestData<double> dd;
    testGetChildFrom<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetChildTo(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_childTo().download(d.m_intNumNonleafNodes);
    EXPECT_EQ(d.m_intNumNonleafNodes[2], 6);
}

TEST_F(MarkovTreeTest, getChildTo) {
    MarkovTreeTestData<float> df;
    testGetChildTo<float>(df);
    MarkovTreeTestData<double> dd;
    testGetChildTo<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetNumChildren(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_numChildren().download(d.m_intNumNonleafNodes);
    EXPECT_EQ(d.m_intNumNonleafNodes[2], 2);
}

TEST_F(MarkovTreeTest, getNumChildren) {
    MarkovTreeTestData<float> df;
    testGetNumChildren<float>(df);
    MarkovTreeTestData<double> dd;
    testGetNumChildren<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetStageFrom(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_stageFrom().download(d.m_intNumStages);
    EXPECT_EQ(d.m_intNumStages[2], 3);
}

TEST_F(MarkovTreeTest, getStageFrom) {
    MarkovTreeTestData<float> df;
    testGetStageFrom<float>(df);
    MarkovTreeTestData<double> dd;
    testGetStageFrom<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testGetStageTo(MarkovTreeTestData<T> &d) {
    d.m_mockTree->d_stageTo().download(d.m_intNumStages);
    EXPECT_EQ(d.m_intNumStages[2], 6);
}

TEST_F(MarkovTreeTest, getStageTo) {
    MarkovTreeTestData<float> df;
    testGetStageTo<float>(df);
    MarkovTreeTestData<double> dd;
    testGetStageTo<double>(dd);
}
