#include <gtest/gtest.h>
#include <numeric>
#include "../src/projections.cuh"
#include "../src/tree.cuh"


class ProjectionsTest : public testing::Test {

protected:
    /* Prepare some host and device data */
    size_t m_n = 520;
    DTensor<DEFAULT_FPX> m_d_data = DTensor<DEFAULT_FPX>(m_n, 1, 1, true);
    std::vector<DEFAULT_FPX> m_data = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_socA = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_socB = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_socC = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_test = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_zero = std::vector<DEFAULT_FPX>(m_n);
    DTensor<DEFAULT_FPX> d_singleProjectSize = DTensor<DEFAULT_FPX>(m_d_data.numRows());
    SocProjection<DEFAULT_FPX> socProj = SocProjection<DEFAULT_FPX>(d_singleProjectSize);

    ProjectionsTest() {
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

    virtual ~ProjectionsTest() {}
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

TEST_F(ProjectionsTest, SocProjI3) {
    m_d_data.upload(m_socC);
    socProj.project(m_d_data);
    m_d_data.download(m_test);
    testSocElse(m_test);
}

TEST_F(ProjectionsTest, SocProjI2) {
    m_d_data.upload(m_socB);
    socProj.project(m_d_data);
    m_d_data.download(m_test);
    EXPECT_EQ(m_test, m_zero);
}

TEST_F(ProjectionsTest, SocProjI1) {
    m_d_data.upload(m_socA);
    socProj.project(m_d_data);
    m_d_data.download(m_test);
    EXPECT_EQ(m_test, m_socA);
}

TEST_F(ProjectionsTest, SocProjI321) {
    /* If the norms of SocProjection are not reset to zeros before each projection,
     * this test will fail.
     */
    m_d_data.upload(m_socC);
    socProj.project(m_d_data);
    m_d_data.download(m_test);
    testSocElse(m_test);
    m_d_data.upload(m_socB);
    socProj.project(m_d_data);
    m_d_data.download(m_test);
    EXPECT_EQ(m_test, m_zero);
    m_d_data.upload(m_socA);
    socProj.project(m_d_data);
    m_d_data.download(m_test);
    EXPECT_EQ(m_test, m_socA);
}

TEST_F(ProjectionsTest, CartesianCone) {
    size_t coneDim = 5;
    size_t numCones = 3;
    std::vector<DEFAULT_FPX> socs = {1., 2., 3., 4., 0.5,
                                     5., 6., 7., 8., -200,
                                     9., -10., 11., -12., 100};
    DTensor<DEFAULT_FPX> d_socs(socs, coneDim, numCones);
    SocProjection multiSocProj(d_socs);
    multiSocProj.project(d_socs);
    std::vector<DEFAULT_FPX> test(coneDim, numCones);
    d_socs.download(test);
    std::vector<DEFAULT_FPX> expected = {0.5456435464587639, 1.0912870929175278, 1.6369306393762917, 2.1825741858350556,
                                         2.988612787525831,
                                         0., 0., 0., 0., 0.,
                                         9., -10., 11., -12., 100.};
    multiSocProj.project(d_socs);
    EXPECT_EQ(test, expected);
}

TEST_F(ProjectionsTest, Serial) {
    size_t coneDim = 5;
    size_t numCones = 3;
    std::vector<DEFAULT_FPX> socs = {1., 2., 3., 4., 0.5,
                                     5., 6., 7., 8., -200,
                                     9., -10., 11., -12., 100};
    std::vector<std::vector<DEFAULT_FPX>> split(numCones);
    for (size_t i = 0; i < numCones; i++) {
        split[i] = std::vector<DEFAULT_FPX>(socs.begin() + coneDim * i,
                                            socs.begin() + coneDim * (i + 1));
    }
    std::vector<std::vector<DEFAULT_FPX>> expected(numCones);
    expected[0] = {0.56681531047810607, 1.1336306209562121, 1.7004459314343183, 2.2672612419124243, 2.1208286933869704};
    expected[1] = {0., 0., 0., 0., 0.};
    expected[2] = {9., -10., 11., -12., 100.};
    for (size_t i = 0; i < numCones; i++) { socProjectSerial(coneDim, split[i]); }
    for (size_t i = 0; i < numCones; i++) { EXPECT_EQ(split[i], expected[i]); }
}


