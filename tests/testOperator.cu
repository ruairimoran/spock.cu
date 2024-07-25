#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <fstream>
#include <filesystem>
#include <iostream>


class OperatorTest : public testing::Test {
protected:
    OperatorTest() {}

    virtual ~OperatorTest() {}
};

TEMPLATE_WITH_TYPE_T
class TestOperatorData {
public:
    std::string m_treeFileLoc = "../../tests/testTreeData.json";
    std::string m_problemFileLoc = "../../tests/testProblemData.json";
    std::unique_ptr<ScenarioTree<T>> m_tree;
    std::unique_ptr<ProblemData<T>> m_data;
    std::unique_ptr<Cache<T>> m_cache;

    /** Prepare some host and device data */
    size_t m_n = 64;
    T m_tol = 1e-4;
    size_t m_maxIters = 20;
    DTensor<T> m_d_data = DTensor<T>(m_n);
    std::vector<T> m_hostData = std::vector<T>(m_n);
    std::vector<T> m_hostTest = std::vector<T>(m_n);

    TestOperatorData() {
        std::ifstream tree_data(m_treeFileLoc);
        std::ifstream problem_data(m_problemFileLoc);
        m_tree = std::make_unique<ScenarioTree<T>>(tree_data);
        m_data = std::make_unique<ProblemData<T>>(*m_tree, problem_data);
        m_cache = std::make_unique<Cache<T>>(*m_tree, *m_data, m_tol, m_maxIters);

        /** Positive and negative values in m_data */
        for (size_t i = 0; i < m_n; i = i + 2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i = 1; i < m_n; i = i + 2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);
    };

    virtual ~TestOperatorData() {}
};

TEMPLATE_WITH_TYPE_T
static void parse(size_t nodeIdx, const rapidjson::Value &value, std::vector<T> &vec) {
    size_t numElements = value.Capacity();
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        vec[nodeIdx * numElements + i] = value[i].GetDouble();
    }
}

/* ---------------------------------------
 * Operator L
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testOperator(TestOperatorData<T> &d) {
    Cache<T> &c = *d.m_cache;
    c.m_L.op(*c.m_d_u, *c.m_d_x, *c.m_d_y, *c.m_d_t, *c.m_d_s,
             *c.m_d_i, *c.m_d_ii, *c.m_d_iii, *c.m_d_iv, *c.m_d_v, *c.m_d_vi);
}

TEST_F(OperatorTest, op) {
    TestOperatorData<float> df;
    testOperator<float>(df);
    TestOperatorData<double> dd;
    testOperator<double>(dd);
}

/* ---------------------------------------
 * Operator L adjoint
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testAdjoint(TestOperatorData<T> &d) {
    Cache<T> &c = *d.m_cache;
    c.m_L.adj(*c.m_d_u, *c.m_d_x, *c.m_d_y, *c.m_d_t, *c.m_d_s,
              *c.m_d_i, *c.m_d_ii, *c.m_d_iii, *c.m_d_iv, *c.m_d_v, *c.m_d_vi);
}

TEST_F(OperatorTest, adj) {
    TestOperatorData<float> df;
    testAdjoint<float>(df);
    TestOperatorData<double> dd;
    testAdjoint<double>(dd);
}
