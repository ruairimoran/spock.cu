#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <fstream>
#include <filesystem>
#include <iostream>


class CacheTest : public testing::Test {
protected:
    CacheTest() {}
    virtual ~CacheTest() {}
};


TEMPLATE_WITH_TYPE_T
class Data {
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

    Data() {
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

    virtual ~Data() {}
};

/* ---------------------------------------
 * Initialise state
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void initialiseState(Data<T> &d) {
    std::vector<T> initialState = {3., 5., 4.};
    d.m_cache->initialiseState(initialState);
    std::vector<T> x(initialState.size());
    DTensor<T> firstState(d.m_cache->states(), 0, 0, d.m_data->numStates() - 1);
    firstState.download(x);
    EXPECT_EQ(x, initialState);
}

TEST_F(CacheTest, initialiseState) {
    Data<float> df;
    initialiseState<float>(df);
    Data<double> dd;
    initialiseState<double>(dd);
}

/* ---------------------------------------
 * Project on dynamics (online)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
static void parse(size_t nodeIdx, const rapidjson::Value &value, std::vector<T> &vec) {
    size_t numElements = value.Capacity();
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        vec[nodeIdx * numElements + i] = value[i].GetDouble();
    }
}

TEMPLATE_WITH_TYPE_T
void dynamicsProjectionOnline(Data<T> &d, T epsilon) {
    std::ifstream problem_data(d.m_problemFileLoc);
    std::string json((std::istreambuf_iterator<char>(problem_data)),
                     std::istreambuf_iterator<char>());
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        std::cerr << "Error parsing problem data JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
        throw std::invalid_argument("Cannot parse problem data JSON file for testing DP");
    }
    size_t statesSize = d.m_data->numStates() * d.m_tree->numNodes();
    size_t inputsSize = d.m_data->numInputs() * d.m_tree->numNonleafNodes();
    std::vector<T> originalStates(statesSize);
    std::vector<T> originalInputs(inputsSize);
    std::vector<T> cvxStates(statesSize);
    std::vector<T> cvxInputs(inputsSize);
    const char *nodeString = nullptr;
    for (size_t i = 0; i < d.m_tree->numNodes(); i++) {
        nodeString = std::to_string(i).c_str();
        parse(i, doc["dpStates"][nodeString], originalStates);
        parse(i, doc["dpProjectedStates"][nodeString], cvxStates);
        if (i < d.m_tree->numNonleafNodes()) {
            parse(i, doc["dpInputs"][nodeString], originalInputs);
            parse(i, doc["dpProjectedInputs"][nodeString], cvxInputs);
        }
    }
    std::vector<T> initialState(d.m_data->numStates());
    d.m_cache->states().upload(originalStates);
    d.m_cache->inputs().upload(originalInputs);
    d.m_cache->projectOnDynamics();
    /* Compare spockStates */
    std::vector<T> spockStates(statesSize);
    d.m_cache->states().download(spockStates);
    for (size_t i = 0; i < statesSize; i++) { EXPECT_NEAR(spockStates[i], cvxStates[i], epsilon); }
    /* Compare inputs */
    std::vector<T> spockInputs(inputsSize);
    d.m_cache->inputs().download(spockInputs);
    for (size_t i = 0; i < inputsSize; i++) { EXPECT_NEAR(spockInputs[i], cvxInputs[i], epsilon); }
}

TEST_F(CacheTest, dynamicsProjectionOnline) {
    Data<float> df;
    dynamicsProjectionOnline<float>(df, TEST_PRECISION_LOW);
    Data<double> dd;
    dynamicsProjectionOnline<double>(dd, TEST_PRECISION_HIGH);
}
