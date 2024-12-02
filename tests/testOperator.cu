#include <gtest/gtest.h>
#include "../src/cache.cuh"
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
    std::string m_file;
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
        m_cache = std::make_unique<Cache<T>>(*m_tree, *m_data, m_tol, m_maxIters);
        m_file = m_tree->path() + m_tree->json();
    };

    virtual ~OperatorTestData() = default;
};

/* ---------------------------------------
 * Operator L
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testOperator(OperatorTestData<T> &d, T epsilon) {
    DTensor<T> primBeforeOp = DTensor<T>::parseFromTextFile(d.m_path + "primBeforeOp", rowMajor);
    DTensor<T> dualAfterOpBeforeAdj = DTensor<T>::parseFromTextFile(d.m_path + "dualAfterOpBeforeAdj", rowMajor);
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
    DTensor<T> dualAfterOpBeforeAdj = DTensor<T>::parseFromTextFile(d.m_path + "dualAfterOpBeforeAdj", rowMajor);
    DTensor<T> primAfterAdj = DTensor<T>::parseFromTextFile(d.m_path + "primAfterAdj", rowMajor);
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

