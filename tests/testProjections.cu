#include <gtest/gtest.h>
#include <numeric>
#include "../src/projections.cuh"
#include "../src/tree.cuh"


class ProjectionsTest : public testing::Test {
protected:
    ProjectionsTest() = default;
};


TEMPLATE_WITH_TYPE_T
class ProjectionsTestData {

public:
    /* Prepare some host and device data */
    size_t m_n = 555;
    DTensor<T> m_d_data = DTensor<T>(m_n, 1, 1, true);
    std::vector<T> m_data = std::vector<T>(m_n);
    std::vector<T> m_socA = std::vector<T>(m_n);
    std::vector<T> m_socB = std::vector<T>(m_n);
    std::vector<T> m_socC = std::vector<T>(m_n);
    std::vector<T> m_test = std::vector<T>(m_n);
    std::vector<T> m_zero = std::vector<T>(m_n);
    DTensor<T> d_singleProjectSize = DTensor<T>(m_n);
    SocProjection<T> socProj = SocProjection<T>(d_singleProjectSize, false);

    ProjectionsTestData() {
        /* For testing i1 */
        for (size_t i = 0; i < m_n - 1; i++) { m_socA[i] = 0.; }
        m_socA[m_n - 1] = 1.;
        /* For testing i2 */
        for (size_t i = 0; i < m_n - 1; i++) { m_socB[i] = 0.; }
        m_socB[m_n - 1] = -1.;
        /* For testing i3 */
        for (size_t i = 0; i < m_n - 1; i++) { m_socC[i] = 1.; }
        m_socC[m_n - 1] = 0.;
    }

    virtual ~ProjectionsTestData() = default;
};

TEMPLATE_WITH_TYPE_T
void socProjectSerial(size_t dim, std::vector<T> &vec) {
    std::vector<T> vecFirstPart(vec.begin(), vec.end() - 1);
    std::vector<T> squares(dim - 1);
    T sum = 0;
    for (size_t i = 0; i < dim - 2; i++) {
        T temp = vecFirstPart[i];
        squares[i] = temp * temp;
        sum += squares[i];
    }
    T nrm = sqrt(sum);
    float vecLastElement = vec[dim - 1];
    if (nrm <= vecLastElement) {
        return;  // Do nothing!
    } else if (nrm <= -vecLastElement) {
        for (size_t i = 0; i < dim; i++) { vec[i] = 0.; }
    } else {
        T scaling = (nrm + vecLastElement) / (2. * nrm);
        for (size_t i = 0; i < dim - 1; i++) { vec[i] *= scaling; }
        vec[dim - 1] = scaling * nrm;
    }
}

TEMPLATE_WITH_TYPE_T
void testSocElse(std::vector<T> testVec) {
    T last = testVec[testVec.size() - 1];
    testVec.pop_back();
    T nrm = std::sqrt(std::inner_product(testVec.begin(), testVec.end(), testVec.begin(), 0.));
    EXPECT_TRUE(nrm <= last);
}

TEMPLATE_WITH_TYPE_T
void testSocProjI3(ProjectionsTestData<T> &d) {
    d.m_d_data.upload(d.m_socC);
    d.socProj.project(d.m_d_data);
    d.m_d_data.download(d.m_test);
    testSocElse(d.m_test);
}

TEST_F(ProjectionsTest, socProjI3) {
    ProjectionsTestData<float> df;
    testSocProjI3<float>(df);
    ProjectionsTestData<double> dd;
    testSocProjI3<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testSocProjI2(ProjectionsTestData<T> &d) {
    d.m_d_data.upload(d.m_socB);
    d.socProj.project(d.m_d_data);
    d.m_d_data.download(d.m_test);
    EXPECT_EQ(d.m_test, d.m_zero);
}

TEST_F(ProjectionsTest, socProjI2) {
    ProjectionsTestData<float> df;
    testSocProjI2<float>(df);
    ProjectionsTestData<double> dd;
    testSocProjI2<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testSocProjI1(ProjectionsTestData<T> &d) {
    d.m_d_data.upload(d.m_socA);
    d.socProj.project(d.m_d_data);
    d.m_d_data.download(d.m_test);
    EXPECT_EQ(d.m_test, d.m_socA);
}

TEST_F(ProjectionsTest, socProjI1) {
    ProjectionsTestData<float> df;
    testSocProjI1<float>(df);
    ProjectionsTestData<double> dd;
    testSocProjI1<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testSocProjI321(ProjectionsTestData<T> &d) {
    /* If the norms of SocProjection are not reset to zeros before each projection,
     * this test will fail.
     */
    d.m_d_data.upload(d.m_socC);
    d.socProj.project(d.m_d_data);
    d.m_d_data.download(d.m_test);
    testSocElse(d.m_test);
    d.m_d_data.upload(d.m_socB);
    d.socProj.project(d.m_d_data);
    d.m_d_data.download(d.m_test);
    EXPECT_EQ(d.m_test, d.m_zero);
    d.m_d_data.upload(d.m_socA);
    d.socProj.project(d.m_d_data);
    d.m_d_data.download(d.m_test);
    EXPECT_EQ(d.m_test, d.m_socA);
}

TEST_F(ProjectionsTest, socProjI321) {
    ProjectionsTestData<float> df;
    testSocProjI321<float>(df);
    ProjectionsTestData<double> dd;
    testSocProjI321<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testCartesianCone(ProjectionsTestData<T> &d, T epsilon) {
    size_t coneDim = 5;
    size_t numCones = 3;
    std::vector<T> socs = {1., 2., 3., 4., 0.5,
                           5., 6., 7., 8., -200,
                           9., -10., 11., -12., 100};
    DTensor<T> d_socs(socs, coneDim, numCones);
    SocProjection multiSocProj(d_socs, false);
    multiSocProj.project(d_socs);
    std::vector<T> test(coneDim, numCones);
    d_socs.download(test);
    std::vector<T> expected = {0.5456435464587639, 1.0912870929175278, 1.6369306393762917, 2.1825741858350556,
                               2.988612787525831,
                               0., 0., 0., 0., 0.,
                               9., -10., 11., -12., 100.};
    for (size_t i = 0; i < coneDim * numCones; i++) { EXPECT_NEAR(test[i], expected[i], epsilon); }
}

TEST_F(ProjectionsTest, cartesianCone) {
    ProjectionsTestData<float> df;
    testCartesianCone<float>(df, TEST_PRECISION_LOW);
    ProjectionsTestData<double> dd;
    testCartesianCone<double>(dd, TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void testSerial(ProjectionsTestData<T> &d, T epsilon) {
    size_t coneDim = 5;
    size_t numCones = 3;
    std::vector<T> socs = {1., 2., 3., 4., 0.5,
                           5., 6., 7., 8., -200,
                           9., -10., 11., -12., 100};
    std::vector<std::vector<T>> split(numCones);
    for (size_t i = 0; i < numCones; i++) {
        split[i] = std::vector<T>(socs.begin() + coneDim * i,
                                  socs.begin() + coneDim * (i + 1));
    }
    std::vector<std::vector<T>> expected(numCones);
    expected[0] = {0.56681531047810607, 1.1336306209562121, 1.7004459314343183, 2.2672612419124243, 2.1208286933869704};
    expected[1] = {0., 0., 0., 0., 0.};
    expected[2] = {9., -10., 11., -12., 100.};
    for (size_t i = 0; i < numCones; i++) { socProjectSerial(coneDim, split[i]); }
    for (size_t i = 0; i < numCones; i++) {
        for (size_t j = 0; j < numCones; j++) { EXPECT_NEAR(split[i][j], expected[i][j], epsilon); }
    }
}

TEST_F(ProjectionsTest, serial) {
    ProjectionsTestData<float> df;
    testSerial<float>(df, TEST_PRECISION_LOW);
    ProjectionsTestData<double> dd;
    testSerial<double>(dd, TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void testCartesianWithMultipleBlocksPerCone(ProjectionsTestData<T> &d) {
    /**
     * This test ensures that the shared memory for batched SOC projections works as intended.
     * The cone's dimensions are greater than any possible block size,
     * so each cone requires at least two blocks of threads.
     * If any block does not get counted,
     * the test will fail (the projection will set the cones to zeros = i2 projection).
     * If all blocks are counted,
     * the test will pass (projection will not set the cones to zeros = i3 projection).
    */
    size_t extra = 5;
    size_t base = 1024;
    size_t coneDim = base + (extra * 2);
    size_t numCones = 2;
    DTensor<T> d_socs(coneDim, numCones);
    SocProjection multiSocProj(d_socs, false);
    DTensor<T> d_cone1(d_socs, 1, 0, 0);
    DTensor<T> d_cone2(d_socs, 1, 1, 1);
    T val = 2.;
    std::vector<T> cone1(coneDim, val);
    std::vector<T> cone2(coneDim, val);
    T lastElement = -sqrt(val * (base + extra));
    cone1[coneDim - 1] = lastElement;
    cone2[coneDim - 1] = lastElement;
    d_cone1.upload(cone1);
    d_cone2.upload(cone2);
    multiSocProj.project(d_socs);
    std::vector<T> test1(coneDim);
    std::vector<T> test2(coneDim);
    d_cone1.download(test1);
    d_cone2.download(test2);
    std::vector<T> notExpected(coneDim, 0.);
    EXPECT_NE(test1, notExpected);
    EXPECT_NE(test2, notExpected);
}

TEST_F(ProjectionsTest, cartesianWithMultipleBlocksPerCone) {
    ProjectionsTestData<float> df;
    testCartesianWithMultipleBlocksPerCone<float>(df);
    ProjectionsTestData<double> dd;
    testCartesianWithMultipleBlocksPerCone<double>(dd);
}
