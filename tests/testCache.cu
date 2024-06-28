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
class CacheData {
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

    CacheData() {
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

    virtual ~CacheData() {}
};

TEMPLATE_WITH_TYPE_T
static void parse(size_t nodeIdx, const rapidjson::Value &value, std::vector<T> &vec) {
    size_t numElements = value.Capacity();
    for (rapidjson::SizeType i = 0; i < numElements; i++) {
        vec[nodeIdx * numElements + i] = value[i].GetDouble();
    }
}

/* ---------------------------------------
 * Initialise state
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void initialisingState(CacheData<T> &d) {
    std::vector<T> initialState = {3., 5., 4.};
    d.m_cache->initialiseState(initialState);
    std::vector<T> x(initialState.size());
    DTensor<T> firstState(d.m_cache->states(), 0, 0, d.m_data->numStates() - 1);
    firstState.download(x);
    EXPECT_EQ(x, initialState);
}

TEST_F(CacheTest, initialisingState) {
    CacheData<float> df;
    initialisingState<float>(df);
    CacheData<double> dd;
    initialisingState<double>(dd);
}

/* ---------------------------------------
 * Project on dynamics (online)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void dynamicsProjectionOnline(CacheData<T> &d, T epsilon) {
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
    CacheData<float> df;
    dynamicsProjectionOnline<float>(df, TEST_PRECISION_LOW);
    CacheData<double> dd;
    dynamicsProjectionOnline<double>(dd, TEST_PRECISION_HIGH);
}

/* ---------------------------------------
 * Project on kernels (online)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void kernelProjectionOnline(CacheData<T> &d, T epsilon) {
    std::ifstream problem_data(d.m_problemFileLoc);
    std::string json((std::istreambuf_iterator<char>(problem_data)),
                     std::istreambuf_iterator<char>());
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        std::cerr << "Error parsing problem data JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
        throw std::invalid_argument("Cannot parse problem data JSON file for testing projection on kernels");
    }
    T hi = 100.;
    T lo = -hi;
    size_t matAxis = 2;
    const char *nodeString = nullptr;
    for (size_t node = 0; node < d.m_tree->numNonleafNodes(); node++) {
        size_t chFr = d.m_tree->childFrom()[node];
        size_t chTo = d.m_tree->childTo()[node];
        size_t numCh = d.m_tree->numChildren()[node];
        size_t actualSizeY = numCh * 2 + 1;
        DTensor<T> yPadded(*(d.m_cache->m_d_y), matAxis, node, node);
        DTensor<T> y(yPadded, 0, 0, actualSizeY - 1);
        DTensor<T> t(*(d.m_cache->m_d_t), matAxis, chFr, chTo);
        DTensor<T> s(*(d.m_cache->m_d_s), matAxis, chFr, chTo);
//        DTensor<T> randY = DTensor<T>::createRandomTensor(actualSizeY, 1, 1, lo, hi);
//        DTensor<T> randT = DTensor<T>::createRandomTensor(1, 1, numCh, lo, hi);
//        DTensor<T> randS = DTensor<T>::createRandomTensor(1, 1, numCh, lo, hi);
        std::vector<T> ones(actualSizeY, 1.);
        std::vector<T> twos(numCh, 2.);
        std::vector<T> threes(numCh, 3.);
        DTensor<T> randY(ones, ones.size(), 1, 1);
        DTensor<T> randT(twos, twos.size(), 1, 1);
        DTensor<T> randS(threes, threes.size(), 1, 1);
        randY.deviceCopyTo(y);
        randT.deviceCopyTo(t);
        randS.deviceCopyTo(s);
    }

    d.m_cache->projectOnKernels();

    for (size_t node = 0; node < d.m_tree->numNonleafNodes(); node++) {
        size_t chFr = d.m_tree->childFrom()[node];
        size_t chTo = d.m_tree->childTo()[node];
        size_t numCh = d.m_tree->numChildren()[node];
        size_t actualSizeY = numCh * 2 + 1;
        DTensor<T> yPadded(*(d.m_cache->m_d_y), matAxis, node, node);
        DTensor<T> y(yPadded, 0, 0, actualSizeY - 1);
        DTensor<T> t(*(d.m_cache->m_d_t), matAxis, chFr, chTo);
        DTensor<T> s(*(d.m_cache->m_d_s), matAxis, chFr, chTo);

        DTensor<T> projected(d.m_data->nullDim());
        DTensor<T> projY(projected, 0, 0, actualSizeY - 1);
        DTensor<T> projT(projected, 0, d.m_cache->m_numY, d.m_cache->m_numY + numCh - 1);
        DTensor<T> projS(projected, 0, d.m_cache->m_numY + d.m_tree->numEvents(), d.m_cache->m_numY + d.m_tree->numEvents() + numCh - 1);
        projT.reshape(1, 1, numCh);
        projS.reshape(1, 1, numCh);
        y.deviceCopyTo(projY);
        t.deviceCopyTo(projT);
        s.deviceCopyTo(projS);

        nodeString = std::to_string(node).c_str();
        rapidjson::Value &risk = doc["risks"][nodeString];
        rapidjson::Value &s2 = risk["S2"];
        size_t numEl = s2.Capacity();
        std::vector<T> s2Vec(numEl);
        parse(0, s2, s2Vec);
        size_t nR = doc["rowsS2"].GetInt();
        DTensor<T> kerConMat(s2Vec, nR, d.m_data->nullDim(), 1, rowMajor);

        std::cout << kerConMat;
        std::cout << projected;
        DTensor<T> shouldBeZeros = kerConMat * projected;
        std::vector<T> result(shouldBeZeros.numEl());
        shouldBeZeros.download(result);
        for (size_t i = 0; i < shouldBeZeros.numEl(); i++) { EXPECT_NEAR(result[i], 0., epsilon); }
    }
}

TEST_F(CacheTest, kernelProjectionOnline) {
    CacheData<float> df;
    kernelProjectionOnline<float>(df, TEST_PRECISION_LOW);
    CacheData<double> dd;
    kernelProjectionOnline<double>(dd, TEST_PRECISION_HIGH);
}
