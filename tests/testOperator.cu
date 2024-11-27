#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <fstream>
#include <filesystem>
#include <iostream>


class OperatorTest : public testing::Test {
protected:
    OperatorTest() = default;

    virtual ~OperatorTest() {}
};

TEMPLATE_WITH_TYPE_T
class OperatorTestData {
public:
    std::string m_dataFile;
    std::unique_ptr<ScenarioTree<T>> m_tree;
    std::unique_ptr<ProblemData<T>> m_data;
    std::unique_ptr<Cache<T>> m_cache;
    T m_tol = 1e-4;
    size_t m_maxIters = 20;
    std::vector<T> m_primBeforeOp;
    std::vector<T> m_dualAfterOpBeforeAdj;
    std::vector<T> m_primAfterAdj;

    OperatorTestData() {
        m_tree = std::make_unique<ScenarioTree<T>>("../../data/");
        m_data = std::make_unique<ProblemData<T>>(*m_tree);
        m_cache = std::make_unique<Cache<T>>(*m_tree, *m_data, m_tol, m_maxIters);
        m_dataFile = m_tree->path() + m_tree->json();
    };

    virtual ~OperatorTestData() = default;
};

/* ---------------------------------------
 * Operator L
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testOperator(OperatorTestData<T> &d, T epsilon) {
    /* Get primal before and dual after operator */
    std::ifstream problemData(d.m_dataFile);
    std::string json((std::istreambuf_iterator<char>(problemData)),
                     std::istreambuf_iterator<char>());
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        err << "[testOperator] Cannot parse problem data JSON file: "
            << std::string(GetParseError_En(doc.GetParseError()));
        throw std::invalid_argument(err.str());
    }
    parseVec(doc["primBeforeOp"], d.m_primBeforeOp);
    parseVec(doc["dualAfterOpBeforeAdj"], d.m_dualAfterOpBeforeAdj);
    /* Load primal and test resulting dual */
    Cache<T> &c = *d.m_cache;
    d.m_cache->m_d_workIteratePrim->upload(d.m_primBeforeOp);
    c.m_L.op(*c.m_d_u, *c.m_d_x, *c.m_d_y, *c.m_d_t, *c.m_d_s,
             *c.m_d_i, *c.m_d_ii, *c.m_d_iii, *c.m_d_iv, *c.m_d_v, *c.m_d_vi);
    std::vector<T> test(c.m_sizeDual);
    c.m_d_workIterateDual->download(test);
    for (size_t i = 0; i < c.m_sizeDual; i++) { EXPECT_NEAR(test[i], d.m_dualAfterOpBeforeAdj[i], epsilon); }
}

TEST_F(OperatorTest, op) {
    OperatorTestData<float> df;
    testOperator<float>(df, TEST_PRECISION_LOW);
    OperatorTestData<double> dd;
    testOperator<double>(dd, TEST_PRECISION_HIGH);
}

/* ---------------------------------------
 * Operator L adjoint
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAdjoint(OperatorTestData<T> &d, T epsilon) {
    /* Get dual before and primal after adjoint */
    std::ifstream problemData(d.m_dataFile);
    std::string json((std::istreambuf_iterator<char>(problemData)),
                     std::istreambuf_iterator<char>());
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        err << "[testAdjoint] Cannot parse problem data JSON file: "
            << std::string(GetParseError_En(doc.GetParseError()));
        throw std::invalid_argument(err.str());
    }
    parseVec(doc["dualAfterOpBeforeAdj"], d.m_dualAfterOpBeforeAdj);
    parseVec(doc["primAfterAdj"], d.m_primAfterAdj);
    /* Load dual and test resulting primal */
    Cache<T> &c = *d.m_cache;
    d.m_cache->m_d_workIterateDual->upload(d.m_dualAfterOpBeforeAdj);
    c.m_L.adj(*c.m_d_u, *c.m_d_x, *c.m_d_y, *c.m_d_t, *c.m_d_s,
              *c.m_d_i, *c.m_d_ii, *c.m_d_iii, *c.m_d_iv, *c.m_d_v, *c.m_d_vi);
    std::vector<T> test(c.m_sizePrim);
    c.m_d_workIteratePrim->download(test);
    for (size_t i = 0; i < c.m_sizePrim; i++) { EXPECT_NEAR(test[i], d.m_primAfterAdj[i], epsilon); }
}

TEST_F(OperatorTest, adj) {
    OperatorTestData<float> df;
    testAdjoint<float>(df, TEST_PRECISION_LOW);
    OperatorTestData<double> dd;
    testAdjoint<double>(dd, TEST_PRECISION_HIGH);
}

/* ---------------------------------------
 * Node-to-node data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyNodeToNode(OperatorTestData<T> &d) {
    size_t nR = 9, nC = 1, nM = 7;
    size_t nCopy = 3;
    size_t srcStart = 6;
    size_t dstStart = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(6, nC, nM, true);
    memCpy(&dst, &src, 0, nM - 1, nCopy, dstStart, srcStart);
    for (size_t mat = 0; mat < nM; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, mat), dst(dstStart + ele, 0, mat));
        }
    }
}

TEST_F(OperatorTest, memCpyNodeToNode) {
    OperatorTestData<float> df;
    testMemCpyNodeToNode<float>(df);
    OperatorTestData<double> dd;
    testMemCpyNodeToNode<double>(dd);
}

/* ---------------------------------------
 * Anc-to-node data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyAncToNode(OperatorTestData<T> &d) {
    size_t nR = 23, nC = 1, nM = 7;
    size_t nCopy = 7;
    size_t srcStart = 6;
    size_t dstStart = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(20, nC, nM, true);
    std::vector<size_t> anc = {0, 0, 0, 1, 1, 2, 2};
    DTensor<size_t> d_anc(anc, nM);
    memCpy(&dst, &src, 1, nM - 1, nCopy, dstStart, srcStart, anc2Node, &d_anc);
    for (size_t mat = 1; mat < nM; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, anc[mat]), dst(dstStart + ele, 0, mat));
        }
    }
}

TEST_F(OperatorTest, memCpyAncToNode) {
    OperatorTestData<float> df;
    testMemCpyAncToNode<float>(df);
    OperatorTestData<double> dd;
    testMemCpyAncToNode<double>(dd);
}

/* ---------------------------------------
 * Anc-to-node data transfer fail
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyFailAncZero(OperatorTestData<T> &d) {
    DTensor<T> dummyT(1);
    DTensor<size_t> dummyU(1);
    ASSERT_ANY_THROW(memCpy(&dummyT, &dummyT, 0, 0, 0, 0, 0, anc2Node, &dummyU));
}

TEST_F(OperatorTest, memCpyFailAncZero) {
    OperatorTestData<float> df;
    testMemCpyFailAncZero<float>(df);
    OperatorTestData<double> dd;
    testMemCpyFailAncZero<double>(dd);
}

/* ---------------------------------------
 * Leaf-to-zeroLeaf data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyLeafToZeroLeaf(OperatorTestData<T> &d) {
    size_t nR = 9, nC = 1, nM = 7;
    size_t nCopy = 3;
    size_t srcStart = 6;
    size_t dstStart = 3;
    size_t numNonleafNodes = 3;
    size_t numLeafNodes = 4;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(6, nC, numLeafNodes, true);
    memCpy(&dst, &src, numNonleafNodes, nM - 1, nCopy, dstStart, srcStart, leaf2ZeroLeaf);
    for (size_t mat = 0; mat < numLeafNodes; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, mat + numNonleafNodes), dst(dstStart + ele, 0, mat));
        }
    }
}

TEST_F(OperatorTest, memCpyLeafToZeroLeaf) {
    OperatorTestData<float> df;
    testMemCpyLeafToZeroLeaf<float>(df);
    OperatorTestData<double> dd;
    testMemCpyLeafToZeroLeaf<double>(dd);
}

/* ---------------------------------------
 * ZeroLeaf-to-Leaf data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyZeroLeafToLeaf(OperatorTestData<T> &d) {
    size_t nR = 9, nC = 1, nM = 7;
    size_t nCopy = 3;
    size_t srcStart = 3;
    size_t dstStart = 6;
    size_t numNonleafNodes = 3;
    size_t numLeafNodes = 4;
    DTensor<T> src = DTensor<T>::createRandomTensor(6, nC, numLeafNodes, -10, 10);
    DTensor<T> dst(nR, nC, nM, true);
    memCpy(&dst, &src, numNonleafNodes, nM - 1, nCopy, dstStart, srcStart, zeroLeaf2Leaf);
    for (size_t mat = 0; mat < numLeafNodes; mat++) {
        for (size_t ele = 0; ele < nCopy; ele++) {
            EXPECT_EQ(src(srcStart + ele, 0, mat), dst(dstStart + ele, 0, mat + numNonleafNodes));
        }
    }
}

TEST_F(OperatorTest, memCpyZeroLeafToLeaf) {
    OperatorTestData<float> df;
    testMemCpyZeroLeafToLeaf<float>(df);
    OperatorTestData<double> dd;
    testMemCpyZeroLeafToLeaf<double>(dd);
}

/* ---------------------------------------
 * Child-to-node data copy
 * - Copy first child's data to node
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyChToNode(OperatorTestData<T> &d) {
    size_t nR = 3, nC = 1, nM = 8;
    size_t numNonleafNodes = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM);
    src.deviceCopyTo(dst);
    std::vector<size_t> chFrom = {1, 3, 5};
    std::vector<size_t> numCh = {2, 2, 3};
    DTensor<size_t> d_chFrom(chFrom, numNonleafNodes);
    DTensor<size_t> d_numCh(numCh, numNonleafNodes);
    size_t ch = 0;
    size_t nodeTo = numNonleafNodes - 1;
    k_memCpyCh2Node<<<nodeTo + 1, TPB>>>(dst.raw(), src.raw(), 0, nodeTo, nR, ch, d_chFrom.raw(), d_numCh.raw(), ch);
    for (size_t mat = 0; mat < numNonleafNodes; mat++) {
        for (size_t ele = 0; ele < nR; ele++) {
            EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, chFrom[mat]));
        }
    }
}

TEST_F(OperatorTest, memCpyChToNode) {
    OperatorTestData<float> df;
    testMemCpyChToNode<float>(df);
    OperatorTestData<double> dd;
    testMemCpyChToNode<double>(dd);
}

/* ---------------------------------------
 * Child-to-node data add
 * - Add second child's data to node
 * - All nodes have second child
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAddChToNodeAll(OperatorTestData<T> &d) {
    size_t nR = 3, nC = 1, nM = 8;
    size_t numNonleafNodes = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM);
    src.deviceCopyTo(dst);
    std::vector<size_t> chFrom = {1, 3, 5};
    std::vector<size_t> numCh = {2, 2, 3};
    DTensor<size_t> d_chFrom(chFrom, numNonleafNodes);
    DTensor<size_t> d_numCh(numCh, numNonleafNodes);
    size_t ch = 1;
    size_t nodeTo = numNonleafNodes - 1;
    k_memCpyCh2Node<<<nodeTo + 1, TPB>>>(dst.raw(), src.raw(), 0, nodeTo, nR, ch, d_chFrom.raw(), d_numCh.raw(), ch);
    for (size_t mat = 0; mat < numNonleafNodes; mat++) {
        for (size_t ele = 0; ele < nR; ele++) {
            EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, mat) + src(ele, 0, chFrom[mat] + ch));
        }
    }
}

TEST_F(OperatorTest, addChToNodeAll) {
    OperatorTestData<float> df;
    testAddChToNodeAll<float>(df);
    OperatorTestData<double> dd;
    testAddChToNodeAll<double>(dd);
}

/* ---------------------------------------
 * Child-to-node data add
 * - Add third child's data to node
 * - Not all nodes have third child
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAddChToNode(OperatorTestData<T> &d) {
    size_t nR = 3, nC = 1, nM = 8;
    size_t numNonleafNodes = 3;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM);
    src.deviceCopyTo(dst);
    std::vector<size_t> chFrom = {1, 3, 5};
    std::vector<size_t> numCh = {2, 2, 3};
    DTensor<size_t> d_chFrom(chFrom, numNonleafNodes);
    DTensor<size_t> d_numCh(numCh, numNonleafNodes);
    size_t ch = 2;
    size_t nodeTo = numNonleafNodes - 1;
    k_memCpyCh2Node<<<nodeTo + 1, TPB>>>(dst.raw(), src.raw(), 0, nodeTo, nR, ch, d_chFrom.raw(), d_numCh.raw(), ch);
    /* Test node 0 and 1 did not change */
    size_t mat;
    for (mat = 0; mat < 2; mat++) {
        for (size_t ele = 0; ele < nR; ele++) {
            EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, mat));
        }
    }
    /* Test node 2 data added */
    mat = 2;
    for (size_t ele = 0; ele < nR; ele++) {
        EXPECT_EQ(dst(ele, 0, mat), src(ele, 0, mat) + src(ele, 0, chFrom[mat] + ch));
    }
}

TEST_F(OperatorTest, addChToNode) {
    OperatorTestData<float> df;
    testAddChToNode<float>(df);
    OperatorTestData<double> dd;
    testAddChToNode<double>(dd);
}
