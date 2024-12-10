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
    size_t m_numLeafNodes;
    size_t m_numStages;
    std::vector<size_t> m_intNumNonleafNodes;
    std::vector<size_t> m_intNumNodes;
    std::vector<T> m_realNumNodes;
    std::vector<size_t> m_intNumStages;

    MarkovTreeTestData() {
        m_mockTree = std::make_unique<ScenarioTree<T>>(m_path);
        m_numNonleafNodes = m_mockTree->numNonleafNodes();
        m_numNodes = m_mockTree->numNodes();
        m_numLeafNodes = m_numNodes - m_numNonleafNodes;
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

/* ---------------------------------------
 * Node-to-node data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyNodeToNode() {
    size_t nR = 33, nC = 1, nM = 35;
    size_t nCopy = 20;
    size_t srcStart = 6;
    size_t dstStart = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM, true);
    memCpyNode2Node(dst, src, 0, nM - 1, nCopy, dstStart, srcStart);
    for (size_t mat = 0; mat < nM; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, mat), dst(dstStart + ele, 0, mat));
        }
    }
}

TEST_F(MarkovTreeTest, memCpyNodeToNode) {
    testMemCpyNodeToNode<float>();
    testMemCpyNodeToNode<double>();
}

/* ---------------------------------------
 * Anc-to-node data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyAncToNode(MarkovTreeTestData<T> &d) {
    size_t nR = 23, nC = 1, nM = 7;
    size_t nCopy = 7;
    size_t srcStart = 6;
    size_t dstStart = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(20, nC, nM, true);
    d.m_mockTree->memCpyAnc2Node(dst, src, 1, nM - 1, nCopy, dstStart, srcStart);
    for (size_t mat = 1; mat < nM; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, d.m_mockTree->d_ancestors()(mat, 0, 0)), dst(dstStart + ele, 0, mat));
        }
    }
}

TEST_F(MarkovTreeTest, memCpyAncToNode) {
    MarkovTreeTestData<float> df;
    testMemCpyAncToNode<float>(df);
    MarkovTreeTestData<double> dd;
    testMemCpyAncToNode<double>(dd);
}

/* ---------------------------------------
 * Anc-to-node data transfer fail
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyFailAncZero(MarkovTreeTestData<T> &d) {
    DTensor<T> dummyT(1);
    DTensor<size_t> dummyU(1);
    ASSERT_ANY_THROW(d.m_mockTree->memCpyAnc2Node(dummyT, dummyT, 0, 0, 0, 0, 0));
}

TEST_F(MarkovTreeTest, memCpyFailAncZero) {
    MarkovTreeTestData<float> df;
    testMemCpyFailAncZero<float>(df);
    MarkovTreeTestData<double> dd;
    testMemCpyFailAncZero<double>(dd);
}

/* ---------------------------------------
 * Leaf-to-zeroLeaf data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyLeafToZero(MarkovTreeTestData<T> &d) {
    size_t nR = 9, nC = 1;
    size_t nCopy = 3;
    size_t srcStart = 6;
    size_t dstStart = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, d.m_numNodes, -10, 10);
    DTensor<T> dst(6, nC, d.m_numLeafNodes, true);
    d.m_mockTree->memCpyLeaf2Zero(dst, src, nCopy, dstStart, srcStart);
    for (size_t mat = 0; mat < d.m_numLeafNodes; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, mat + d.m_numNonleafNodes), dst(dstStart + ele, 0, mat));
        }
    }
}

TEST_F(MarkovTreeTest, memCpyLeafToZeroLeaf) {
    MarkovTreeTestData<float> df;
    testMemCpyLeafToZero<float>(df);
    MarkovTreeTestData<double> dd;
    testMemCpyLeafToZero<double>(dd);
}

/* ---------------------------------------
 * ZeroLeaf-to-Leaf data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyZeroToLeaf(MarkovTreeTestData<T> &d) {
    size_t nR = 9, nC = 1;
    size_t nCopy = 3;
    size_t srcStart = 3;
    size_t dstStart = 6;
    DTensor<T> src = DTensor<T>::createRandomTensor(6, nC, d.m_numLeafNodes, -10, 10);
    DTensor<T> dst(nR, nC, d.m_numNodes, true);
    d.m_mockTree->memCpyZero2Leaf(dst, src, nCopy, dstStart, srcStart);
    for (size_t mat = 0; mat < d.m_numLeafNodes; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, mat), dst(dstStart + ele, 0, mat + d.m_numNonleafNodes));
        }
    }
}

TEST_F(MarkovTreeTest, memCpyZeroLeafToLeaf) {
    MarkovTreeTestData<float> df;
    testMemCpyZeroToLeaf<float>(df);
    MarkovTreeTestData<double> dd;
    testMemCpyZeroToLeaf<double>(dd);
}

/* ---------------------------------------
 * Child-to-node data copy
 * - Copy first child's data to node
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyChToNode(MarkovTreeTestData<T> &d) {
    size_t nR = 3, nC = 1, nM = d.m_numNodes;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM);
    src.deviceCopyTo(dst);
    size_t ch = 0;
    d.m_mockTree->memCpyCh2Node(dst, src, 0, d.m_mockTree->numNonleafNodesMinus1(), ch, ch);
    for (size_t mat = 0; mat < d.m_numNonleafNodes; mat++) {
        for (size_t ele = 0; ele < nR; ele++) {
            EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, d.m_mockTree->childFrom()[mat]));
        }
    }
}

TEST_F(MarkovTreeTest, memCpyChToNode) {
    MarkovTreeTestData<float> df;
    testMemCpyChToNode<float>(df);
    MarkovTreeTestData<double> dd;
    testMemCpyChToNode<double>(dd);
}

/* ---------------------------------------
 * Child-to-node data add
 * - Add second child's data to node
 * - Not all nodes have second child
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAddChToNode(MarkovTreeTestData<T> &d) {
    size_t nR = 3, nC = 1, nM = d.m_numNodes;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM);
    src.deviceCopyTo(dst);
    size_t ch = 1;
    d.m_mockTree->memCpyCh2Node(dst, src, 0, d.m_mockTree->numNonleafNodesMinus1(), ch, ch);
    /* Test nodes 0-2 data added */
    for (size_t mat = 0; mat < 3; mat++) {
        for (size_t ele = 0; ele < nR; ele++) {
            EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, mat) + src(ele, 0, d.m_mockTree->childFrom()[mat] + ch));
        }
    }
    /* Test nodes 3-numNonleaf did not change */
    for (size_t mat = 3; mat < d.m_numNonleafNodes; mat++) {
        for (size_t ele = 0; ele < nR; ele++) {
            EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, mat));
        }
    }
}

TEST_F(MarkovTreeTest, memAddChToNode) {
    MarkovTreeTestData<float> df;
    testAddChToNode<float>(df);
    MarkovTreeTestData<double> dd;
    testAddChToNode<double>(dd);
}
