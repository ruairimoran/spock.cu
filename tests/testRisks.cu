#include <gtest/gtest.h>
#include "../src/risks.cuh"
#include "../src/tree.cuh"


class RisksTest : public testing::Test {
protected:
    RisksTest() = default;
};


TEMPLATE_WITH_TYPE_T
class RisksTestData {

public:
    std::string m_path = "../../data/";
    std::unique_ptr<ScenarioTree<T>> m_tree;

    RisksTestData() {
        m_tree = std::make_unique<ScenarioTree<T>>(m_path);
    }

    virtual ~RisksTestData() = default;
};

TEMPLATE_WITH_TYPE_T
void testCreateAvar(RisksTestData<T> &d) {
    AVaR<T> myRisk(d.m_path, d.m_tree->fpFileExt(), d.m_tree->numChildren());
}

TEST_F(RisksTest, createAvar) {
    RisksTestData<float> df;
    testCreateAvar<float>(df);
    RisksTestData<double> dd;
    testCreateAvar<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testIndexedNnoc(RisksTestData<T> &d) {
    AVaR<T> avar(d.m_path, d.m_tree->fpFileExt(), d.m_tree->numChildren());
    size_t dim = avar.dimension();
    DTensor<T> d_vec(std::vector<T>(dim, -10.), dim);
    avar.projectDual(d_vec);
    std::vector<T> test(dim);
    d_vec.download(test);
    std::vector<T> expected(dim, -0.);
    std::vector<size_t> idx = {4, 9, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 32, 33, 34};
    for (size_t i : idx) { expected[i] = -10.; }
    EXPECT_EQ(test, expected);
}

TEST_F(RisksTest, indexedNnoc) {
    RisksTestData<float> df;
    testIndexedNnoc<float>(df);
    RisksTestData<double> dd;
    testIndexedNnoc<double>(dd);
}
