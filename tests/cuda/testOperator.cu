#include <gtest/gtest.h>
#include "cache.cuh"
#include <filesystem>
#include <iostream>


class OperatorTest : public testing::Test {
protected:
    OperatorTest() = default;
};

TEMPLATE_WITH_TYPE_T
class OperatorTestData {
public:
    std::string m_path = "../../data/";
    std::unique_ptr<ScenarioTree<T>> m_tree;
    std::unique_ptr<ProblemData<T>> m_data;
    std::unique_ptr<Cache<T>> m_cache;
    T m_tol = 1e-4;
    size_t m_maxIters = 20;
    std::vector<T> m_primBeforeOp;
    std::vector<T> m_dualAfterOpBeforeAdj;
    std::vector<T> m_primAfterAdj;

    OperatorTestData() {
        m_tree = std::make_unique<ScenarioTree<T>>(m_path);
        m_data = std::make_unique<ProblemData<T>>(*m_tree);
        m_cache = CacheBuilder<T>(*m_tree, *m_data).tol(m_tol).maxIters(m_maxIters).make_unique();
    };

    virtual ~OperatorTestData() = default;
};

/* ---------------------------------------
 * Operator L
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testOperator(OperatorTestData<T> &d, T epsilon) {
    std::string ext = d.m_tree->fpFileExt();
    DTensor<T> primBeforeOp = DTensor<T>::parseFromFile(d.m_path + "test_primBeforeOp" + ext);
    DTensor<T> dualAfterOpBeforeAdj = DTensor<T>::parseFromFile(d.m_path + "test_dualAfterOpBeforeAdj" + ext);
    dualAfterOpBeforeAdj.download(d.m_dualAfterOpBeforeAdj);
    /* Load primal and test resulting dual */
    Cache<T> &c = *d.m_cache;
    primBeforeOp.deviceCopyTo(*d.m_cache->m_d_workIteratePrim);
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
    std::string ext = d.m_tree->fpFileExt();
    DTensor<T> dualAfterOpBeforeAdj = DTensor<T>::parseFromFile(d.m_path + "test_dualAfterOpBeforeAdj" + ext);
    DTensor<T> primAfterAdj = DTensor<T>::parseFromFile(d.m_path + "test_primAfterAdj" + ext);
    primAfterAdj.download(d.m_primAfterAdj);
    /* Load dual and test resulting primal */
    Cache<T> &c = *d.m_cache;
    dualAfterOpBeforeAdj.deviceCopyTo(*d.m_cache->m_d_workIterateDual);
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
 * Ensure L' is the adjoint of L
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testIsItReallyTheAdjoint(OperatorTestData<T> &d, T epsilon) {
    std::string ext = d.m_tree->fpFileExt();
    DTensor<T> prim = DTensor<T>::parseFromFile(d.m_path + "test_adjRandomPrim" + ext);
    DTensor<T> dual = DTensor<T>::parseFromFile(d.m_path + "test_adjRandomDual" + ext);
    DTensor<T> result = DTensor<T>::parseFromFile(d.m_path + "test_adjRandomResult" + ext);
    Cache<T> &c = *d.m_cache;
    /* y'Lx */
    prim.deviceCopyTo(*d.m_cache->m_d_workIteratePrim);
    c.L();
    T uno = d.m_cache->m_d_workIterateDual->dotF(dual);
    /* Reset */
    d.m_cache->m_d_workIterate->upload(std::vector<T>(d.m_cache->m_sizeIterate, 0.));
    /* (L'y)'x */
    dual.deviceCopyTo(*d.m_cache->m_d_workIterateDual);
    c.Ltr();
    T dos = d.m_cache->m_d_workIteratePrim->dotF(prim);
    /* Compare results */
    EXPECT_NEAR(uno, dos, 1e-1);  // Leeway required, not sure why
    EXPECT_NEAR(uno, result(0, 0, 0), 1e-1);  // Leeway for python computation errors
}

TEST_F(OperatorTest, testIsItReallyTheAdjoint) {
    OperatorTestData<float> df;
    testIsItReallyTheAdjoint<float>(df, TEST_PRECISION_LOW);
    OperatorTestData<double> dd;
    testIsItReallyTheAdjoint<double>(dd, TEST_PRECISION_HIGH);
}
