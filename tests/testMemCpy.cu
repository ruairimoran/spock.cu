#include <gtest/gtest.h>
#include "../src/memCpy.cuh"


class MemCpyTest : public testing::Test {
protected:
    MemCpyTest() = default;
};


/* ---------------------------------------
 * Node-to-node data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyNodeToNode() {
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

TEST_F(MemCpyTest, memCpyNodeToNode) {
    testMemCpyNodeToNode<float>();
    testMemCpyNodeToNode<double>();
}

/* ---------------------------------------
 * Anc-to-node data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyAncToNode() {
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

TEST_F(MemCpyTest, memCpyAncToNode) {
    testMemCpyAncToNode<float>();
    testMemCpyAncToNode<double>();
}

/* ---------------------------------------
 * Anc-to-node data transfer fail
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyFailAncZero() {
    DTensor<T> dummyT(1);
    DTensor<size_t> dummyU(1);
    ASSERT_ANY_THROW(memCpy(&dummyT, &dummyT, 0, 0, 0, 0, 0, anc2Node, &dummyU));
}

TEST_F(MemCpyTest, memCpyFailAncZero) {
    testMemCpyFailAncZero<float>();
    testMemCpyFailAncZero<double>();
}

/* ---------------------------------------
 * Leaf-to-zeroLeaf data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyLeafToZeroLeaf() {
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

TEST_F(MemCpyTest, memCpyLeafToZeroLeaf) {
    testMemCpyLeafToZeroLeaf<float>();
    testMemCpyLeafToZeroLeaf<double>();
}

/* ---------------------------------------
 * ZeroLeaf-to-Leaf data transfer
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyZeroLeafToLeaf() {
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

TEST_F(MemCpyTest, memCpyZeroLeafToLeaf) {
    testMemCpyZeroLeafToLeaf<float>();
    testMemCpyZeroLeafToLeaf<double>();
}

/* ---------------------------------------
 * Child-to-node data copy
 * - Copy first child's data to node
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyChToNode() {
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

TEST_F(MemCpyTest, memCpyChToNode) {
    testMemCpyChToNode<float>();
    testMemCpyChToNode<double>();
}

/* ---------------------------------------
 * Child-to-node data add
 * - Add second child's data to node
 * - All nodes have second child
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAddChToNodeAll() {
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

TEST_F(MemCpyTest, addChToNodeAll) {
    testAddChToNodeAll<float>();
    testAddChToNodeAll<double>();
}

/* ---------------------------------------
 * Child-to-node data add
 * - Add third child's data to node
 * - Not all nodes have third child
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAddChToNode() {
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

TEST_F(MemCpyTest, addChToNode) {
    testAddChToNode<float>();
    testAddChToNode<double>();
}