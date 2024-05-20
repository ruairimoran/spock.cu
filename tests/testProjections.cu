#include <gtest/gtest.h>
#include <numeric>
#include "../src/projections.cuh"
#include "../src/tree.cuh"


class ProjectionsTest : public testing::Test {

protected:
    /** Prepare some host and device data */
    size_t m_n = 64;
    size_t m_numConeTypes = 4;
    DTensor<DEFAULT_FPX> m_d_data = DTensor<DEFAULT_FPX>(m_n);
    DTensor<DEFAULT_FPX> m_d_dataCart = DTensor<DEFAULT_FPX>(m_n * m_numConeTypes);
    std::vector<DEFAULT_FPX> m_hostData = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostTest = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostZero = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostSocA = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostSocB = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostSocC = std::vector<DEFAULT_FPX>(m_n);
    std::vector<DEFAULT_FPX> m_hostCart = std::vector<DEFAULT_FPX>(m_n * m_numConeTypes);
    std::vector<DEFAULT_FPX> m_testCart;

    ProjectionsTest() {
        /** Positive and negative values in m_hostData */
        for (size_t i = 0; i < m_n; i = i + 2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i = 1; i < m_n; i = i + 2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);  ///< Main vector for projection testing
        /** Zeroes in m_hostZero */
        for (size_t i = 0; i < m_n; i++) { m_hostZero[i] = 0.; }
        /** For testing `if` projection of SOC */
        for (size_t i = 0; i < m_n - 1; i++) { m_hostSocA[i] = 0.; }
        m_hostSocA[m_n - 1] = 1.;
        /** For testing `else if` projection of SOC */
        for (size_t i = 0; i < m_n - 1; i++) { m_hostSocB[i] = 0.; }
        m_hostSocB[m_n - 1] = -1.;
        /** For testing `else` projection of SOC */
        for (size_t i = 0; i < m_n - 1; i++) { m_hostSocC[i] = 1.; }
        m_hostSocC[m_n - 1] = 0.;
        /** For testing Cartesian cone of all types */
        for (size_t i = 0; i < m_n * m_numConeTypes; i++) {
            if (m_n * 0 <= i && i < m_n * 1) m_hostCart[i] = m_hostData[i % m_n];  ///< For projecting Cartesian::Univ
            if (m_n * 1 <= i && i < m_n * 2) m_hostCart[i] = m_hostData[i % m_n];  ///< For projecting Cartesian::Zero
            if (m_n * 2 <= i && i < m_n * 3) m_hostCart[i] = m_hostData[i % m_n];  ///< For projecting Cartesian::NnOC
            if (m_n * 3 <= i && i < m_n * 4) m_hostCart[i] = m_hostSocC[i % m_n];  ///< For projecting Cartesian::SOC
        }
    }

    virtual ~ProjectionsTest() {}
};

TEMPLATE_WITH_TYPE_T
void testNnocProjection(std::vector<T> testVec) {
    for (size_t i=0; i<testVec.size(); i++) { EXPECT_TRUE(testVec[i] >= 0.); }
}

TEMPLATE_WITH_TYPE_T
void testSocElse(std::vector<T> testVec) {
    T last = testVec[testVec.size() - 1];
    testVec.pop_back();
    T nrm = std::sqrt(std::inner_product(testVec.begin(), testVec.end(), testVec.begin(), 0.));
    EXPECT_TRUE(nrm <= last);
}

TEST_F(ProjectionsTest, SecondOrderConeCone) {
//    SecondOrderCone myCone(m_n);
//    /** Testing `if` projection of SOC */
//    m_d_data.upload(m_hostSocA);
//    myCone.project(m_d_data);
//    m_d_data.download(m_hostTest);
//    EXPECT_TRUE((m_hostTest == m_hostSocA));
//    /** Testing `else if` projection of SOC */
//    m_d_data.upload(m_hostSocB);
//    myCone.project(m_d_data);
//    m_d_data.download(m_hostTest);
//    EXPECT_TRUE((m_hostTest == m_hostZero));
//    /** Testing `else` projection of SOC */
//    m_d_data.upload(m_hostSocC);
//    myCone.project(m_d_data);
//    m_d_data.download(m_hostTest);
//    testSocElse(m_hostTest);
}

TEST_F(ProjectionsTest, CartesianCone) {
//    // for (size_t i=0; i<m_n*m_numConeTypes; i++) { std::cerr << m_hostCart[i] << " "; }  ///< For debugging
//    m_d_dataCart.upload(m_hostCart);
//    UniverseCone myUniv(m_n);
//    ZeroCone myZero(m_n);
//    NonnegativeOrthantCone myNnoc(m_n);
//    SecondOrderCone mySoc(m_n);
//    Cartesian myCone;
//    myCone.addCone(myUniv);
//    myCone.addCone(myZero);
//    myCone.addCone(myNnoc);
//    myCone.addCone(mySoc);
//    myCone.project(m_d_dataCart);
//    m_d_dataCart.download(m_hostCart);
//    /** Test Universe cone */
//    size_t index = 0;
//    m_testCart = std::vector<DEFAULT_FPX>(m_hostCart.begin() + index, m_hostCart.begin() + index + m_n);
//    EXPECT_TRUE((m_testCart == m_hostData));
//    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
//    /** Test Zero cone */
//    index += m_n;
//    m_testCart = std::vector<DEFAULT_FPX>(m_hostCart.begin() + index, m_hostCart.begin() + index + m_n);
//    EXPECT_TRUE((m_testCart == m_hostZero));
//    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
//    /** Test NnOC cone */
//    index += m_n;
//    m_testCart = std::vector<DEFAULT_FPX>(m_hostCart.begin() + index, m_hostCart.begin() + index + m_n);
//    testNnocProjection(m_testCart);
//    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
//    /** Test SOC cone */
//    index += m_n;
//    m_testCart = std::vector<DEFAULT_FPX>(m_hostCart.begin() + index, m_hostCart.begin() + index + m_n);
//    testSocElse(m_testCart);
//    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
}


