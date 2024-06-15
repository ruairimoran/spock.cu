#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <fstream>
#include <filesystem>
#include <iostream>


class CacheTest : public testing::Test {

protected:
    std::string m_treeFileLoc = "../../tests/testTreeData.json";
    std::string m_problemFileLoc = "../../tests/testProblemData.json";
    std::unique_ptr<ScenarioTree<DEFAULT_FPX>> m_tree;
    std::unique_ptr<ProblemData<DEFAULT_FPX>> m_data;
    std::unique_ptr<Cache<DEFAULT_FPX>> m_cache;

    /** Prepare some host and device data */
    size_t m_n = 64;
    DEFAULT_FPX m_tol = 1e-4;
    size_t m_maxIters = 20;
    DTensor<DEFAULT_FPX> m_d_data = DTensor<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostData = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostTest = std::vector<DEFAULT_FPX>(m_n);

    CacheTest() {
        std::ifstream tree_data(m_treeFileLoc);
        std::ifstream problem_data(m_problemFileLoc);
        m_tree = std::make_unique<ScenarioTree<DEFAULT_FPX>>(tree_data);
        m_data = std::make_unique<ProblemData<DEFAULT_FPX>>(*m_tree, problem_data);
        m_cache = std::make_unique<Cache<DEFAULT_FPX>>(*m_tree, *m_data, m_tol, m_maxIters);

        /** Positive and negative values in m_data */
        for (size_t i = 0; i < m_n; i = i + 2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i = 1; i < m_n; i = i + 2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);
    };

    virtual ~CacheTest() {}
};

template<typename T>
static void parse(size_t nodeIdx, const rapidjson::Value &value, std::vector<T> &vec) {
    size_t numElements = value.Capacity();
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        vec[nodeIdx * numElements + i] = value[i].GetDouble();
    }
}

TEST_F(CacheTest, InitialiseState) {
    std::vector<DEFAULT_FPX> initialState = {3., 5., 4.};
    m_cache->initialiseState(initialState);
    std::vector<DEFAULT_FPX> sol(m_cache->solutionSize());
    m_cache->solution().download(sol);
    EXPECT_EQ(sol, initialState);
}

TEST_F(CacheTest, DynamicsProjectionOnline) {
    std::ifstream problem_data(m_problemFileLoc);
    std::string json((std::istreambuf_iterator<char>(problem_data)),
                     std::istreambuf_iterator<char>());
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        std::cerr << "Error parsing problem data JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
        throw std::invalid_argument("Cannot parse problem data JSON file for testing DP");
    }
    size_t statesSize = m_data->numStates() * m_tree->numNodes();
    size_t inputsSize = m_data->numInputs() * m_tree->numNonleafNodes();
    std::vector<DEFAULT_FPX> cvxStates(statesSize);
    std::vector<DEFAULT_FPX> cvxInputs(inputsSize);
    const char *nodeString = nullptr;
    for (size_t i = 0; i < m_tree->numNodes(); i++) {
        nodeString = std::to_string(i).c_str();
        parse(i, doc["dpStates"][nodeString], cvxStates);
        if (i < m_tree->numNonleafNodes()) parse(i, doc["dpInputs"][nodeString], cvxInputs);
    }
    std::vector<DEFAULT_FPX> initialState(m_data->numStates());
    parse(0, doc["dpStates"]["0"], initialState);
    m_cache->initialiseState(initialState);
    m_cache->projectOnDynamics();
    /* Compare spockStates */
    std::vector<DEFAULT_FPX> spockStates(statesSize);
    m_cache->states().download(spockStates);
    EXPECT_EQ(spockStates, cvxStates);
    /* Compare inputs */
    std::vector<DEFAULT_FPX> spockInputs(inputsSize);
    m_cache->inputs().download(spockInputs);
    EXPECT_EQ(spockInputs, cvxInputs);
}
