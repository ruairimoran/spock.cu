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
    std::string m_treeFileLoc = std::string("../../tests/testTreeData.json");
    std::string m_problemFileLoc = std::string("../../tests/testProblemData.json");
    std::unique_ptr<ScenarioTree<T>> m_tree;
    std::unique_ptr<ProblemData<T>> m_data;
    std::unique_ptr<Cache<T>> m_cache;
    T m_tol = 1e-4;
    size_t m_maxIters = 20;
    std::vector<T> m_primBeforeOp;
    std::vector<T> m_dualAfterOpBeforeAdj;
    std::vector<T> m_primAfterAdj;

    OperatorTestData() {
        std::ifstream treeData(m_treeFileLoc);
        std::ifstream problemData(m_problemFileLoc);
        m_tree = std::make_unique<ScenarioTree<T>>(treeData);
        m_data = std::make_unique<ProblemData<T>>(*m_tree, problemData);
        m_cache = std::make_unique<Cache<T>>(*m_tree, *m_data, m_tol, m_maxIters);
    };

    virtual ~OperatorTestData() = default;
};

TEMPLATE_WITH_TYPE_T
static void parseVec(const rapidjson::Value &value, std::vector<T> &vec) {
    size_t numElements = value.Capacity();
    if (vec.capacity() != numElements) vec.resize(numElements);
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        vec[i] = value[i].GetDouble();
    }
}

/* ---------------------------------------
 * Operator L
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testOperator(OperatorTestData<T> &d, T epsilon) {
    /* Get primal before and dual after operator */
    std::ifstream problemData(d.m_problemFileLoc);
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
    d.m_cache->m_d_primWorkspace->upload(d.m_primBeforeOp);
    c.m_L.op(*c.m_d_u, *c.m_d_x, *c.m_d_y, *c.m_d_t, *c.m_d_s,
             *c.m_d_i, *c.m_d_ii, *c.m_d_iii, *c.m_d_iv, *c.m_d_v, *c.m_d_vi);
    std::vector<T> test(c.m_dualSize);
    c.m_d_dualWorkspace->download(test);
    for (size_t i = 0; i < c.m_dualSize; i++) { EXPECT_NEAR(test[i], d.m_dualAfterOpBeforeAdj[i], epsilon); }
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
    std::ifstream problemData(d.m_problemFileLoc);
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
    d.m_cache->m_d_dualWorkspace->upload(d.m_dualAfterOpBeforeAdj);
    c.m_L.adj(*c.m_d_u, *c.m_d_x, *c.m_d_y, *c.m_d_t, *c.m_d_s,
              *c.m_d_i, *c.m_d_ii, *c.m_d_iii, *c.m_d_iv, *c.m_d_v, *c.m_d_vi);
    std::vector<T> test(c.m_primSize);
    c.m_d_primWorkspace->download(test);
    for (size_t i = 0; i < c.m_primSize; i++) { EXPECT_NEAR(test[i], d.m_primAfterAdj[i], epsilon); }
}

TEST_F(OperatorTest, adj) {
    OperatorTestData<float> df;
    testAdjoint<float>(df, TEST_PRECISION_LOW);
    OperatorTestData<double> dd;
    testAdjoint<double>(dd, TEST_PRECISION_HIGH);
}
