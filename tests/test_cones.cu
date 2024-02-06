#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include "../src/cones.cuh"


class ConesTest : public testing::Test {
	
    protected:
        Context context;  ///< Create one context only

        /** Prepare some host and device data */
        size_t m_n = 64;
        size_t m_numConeTypes = 4;
        DeviceVector<real_t> m_d_data = DeviceVector<real_t>(m_n);
        std::vector<real_t> m_hostData = std::vector<real_t>(m_n);
        std::vector<real_t> m_hostTest = std::vector<real_t>(m_n);
        std::vector<real_t> m_hostZero = std::vector<real_t>(m_n);
        std::vector<real_t> m_hostSocA = std::vector<real_t>(m_n);
        std::vector<real_t> m_hostSocB = std::vector<real_t>(m_n);
        std::vector<real_t> m_hostSocC = std::vector<real_t>(m_n);
        std::vector<real_t> m_hostCart = std::vector<real_t>(m_n * m_numConeTypes);
        std::vector<real_t> m_testCart;
        ConesTest() {
            /** Positive and negative values in m_hostData */
            for (size_t i=0; i<m_n; i=i+2) { m_hostData[i] = -2. * (i + 1.); }
            for (size_t i=1; i<m_n; i=i+2) { m_hostData[i] = 2. * (i + 1.); }
            m_d_data.upload(m_hostData);  ///< Main vector for projection testing
            /** Zeroes in m_hostZero */
            for (size_t i=0; i<m_n; i++) { m_hostZero[i] = 0.; }
            /** For testing `if` projection of SOC */
            for (size_t i=0; i<m_n-1; i++) { m_hostSocA[i] = 0.; }
            m_hostSocA[m_n-1] = 1.;
            /** For testing `else if` projection of SOC */
            for (size_t i=0; i<m_n-1; i++) { m_hostSocB[i] = 0.; }
            m_hostSocB[m_n-1] = -1.;
            /** For testing `else` projection of SOC */
            for (size_t i=0; i<m_n-1; i++) { m_hostSocC[i] = 1.; }
            m_hostSocC[m_n-1] = 0.;
            /** For testing Cartesian cone of all types */
            for (size_t i=0; i<m_n*m_numConeTypes; i++) {
                if (m_n * 0 <= i && i < m_n * 1) m_hostCart[i] = m_hostData[i%m_n];  ///< For projecting Cartesian::Univ
                if (m_n * 1 <= i && i < m_n * 2) m_hostCart[i] = m_hostData[i%m_n];  ///< For projecting Cartesian::Zero
                if (m_n * 2 <= i && i < m_n * 3) m_hostCart[i] = m_hostData[i%m_n];  ///< For projecting Cartesian::NnOC
                if (m_n * 3 <= i && i < m_n * 4) m_hostCart[i] = m_hostSocC[i%m_n];  ///< For projecting Cartesian::SOC
            }
        };

        virtual ~ConesTest() {}
};


void testNnocProjection(std::vector<real_t> testVec) {
    for (size_t i=0; i<testVec.size(); i++) { EXPECT_TRUE(testVec[i] >= 0.); }
}


void testSocElse(std::vector<real_t> testVec) {
    real_t last = testVec[testVec.size() - 1];
    testVec.pop_back();
    real_t nrm = std::sqrt(std::inner_product(testVec.begin(), testVec.end(), testVec.begin(), 0.));
    EXPECT_TRUE(nrm <= last);
}


TEST_F(ConesTest, UniverseCone) {
    UniverseCone myCone(context);
    myCone.projectOnCone(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostData));
}

TEST_F(ConesTest, UniverseDual) {
    UniverseCone myCone(context);
    myCone.projectOnDual(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostZero));
}

TEST_F(ConesTest, ZeroCone) {
    ZeroCone myCone(context);
    myCone.projectOnCone(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostZero));
}

TEST_F(ConesTest, ZeroDual) {
    ZeroCone myCone(context);
    myCone.projectOnDual(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostData));
}

TEST_F(ConesTest, NonnegativeOrthantCone) {
    NonnegativeOrthantCone myCone(context);
    myCone.projectOnCone(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    testNnocProjection(m_hostTest);
}

TEST_F(ConesTest, NonnegativeOrthantDual) {
    NonnegativeOrthantCone myCone(context);
    myCone.projectOnDual(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    testNnocProjection(m_hostTest);
}

TEST_F(ConesTest, SecondOrderConeCone) {
    SecondOrderCone myCone(context);
    /** Testing `if` projection of SOC */
    m_d_data.upload(m_hostSocA);
    myCone.projectOnCone(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostSocA));
    /** Testing `else if` projection of SOC */
    m_d_data.upload(m_hostSocB);
    myCone.projectOnCone(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostZero));
    /** Testing `else` projection of SOC */
    m_d_data.upload(m_hostSocC);
    myCone.projectOnCone(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    testSocElse(m_hostTest);
}

TEST_F(ConesTest, SecondOrderConeDual) {
    SecondOrderCone myCone(context);
    /** Testing `if` projection of SOC */
    m_d_data.upload(m_hostSocA);
    myCone.projectOnDual(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostSocA));
    /** Testing `else if` projection of SOC */
    m_d_data.upload(m_hostSocB);
    myCone.projectOnDual(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    EXPECT_TRUE((m_hostTest == m_hostZero));
    /** Testing `else` projection of SOC */
    m_d_data.upload(m_hostSocC);
    myCone.projectOnDual(m_d_data.get(), m_d_data.capacity());
    m_d_data.download(m_hostTest);
    testSocElse(m_hostTest);
}

TEST_F(ConesTest, CartesianCone) {
    // for (size_t i=0; i<m_n*m_numConeTypes; i++) { std::cerr << m_hostCart[i] << " "; }  ///< For debugging
    m_d_data.upload(m_hostCart);
    UniverseCone myUniverseCone(context);
    ZeroCone myZeroCone(context);
    NonnegativeOrthantCone myNnocCone(context);
    SecondOrderCone mySocCone(context);
    std::vector<ConvexCone*> myConesVec {&myUniverseCone, &myZeroCone, &myNnocCone, &mySocCone};
    std::vector<size_t> mySizesVec(m_numConeTypes, m_n);
    Cartesian myCone(context, myConesVec, mySizesVec);
    myCone.projectOnCone(m_d_data.get());
    m_d_data.download(m_hostCart);
    /** Test Universe cone */
    size_t index = 0;
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[0]);
    EXPECT_TRUE((m_testCart == m_hostData));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test Zero cone */
    index += mySizesVec[0];
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[1]);
    EXPECT_TRUE((m_testCart == m_hostZero));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test NnOC cone */
    index += mySizesVec[1];
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[2]);
    testNnocProjection(m_testCart);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test SOC cone */
    index += mySizesVec[2];
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[3]);
    testSocElse(m_testCart);
    // for (size_t i=0; i<n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
}

TEST_F(ConesTest, CartesianDual) {
    // for (size_t i=0; i<m_n*m_numConeTypes; i++) { std::cerr << m_hostCart[i] << " "; }  ///< For debugging
    m_d_data.upload(m_hostCart);
    UniverseCone myUniverseCone(context);
    ZeroCone myZeroCone(context);
    NonnegativeOrthantCone myNnocCone(context);
    SecondOrderCone mySocCone(context);
    std::vector<ConvexCone*> myConesVec {&myUniverseCone, &myZeroCone, &myNnocCone, &mySocCone};
    std::vector<size_t> mySizesVec(m_numConeTypes, m_n);
    Cartesian myCone(context, myConesVec, mySizesVec);
    myCone.projectOnDual(m_d_data.get());
    m_d_data.download(m_hostCart);
    /** Test Universe dual */
    size_t index = 0;
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[0]);
    EXPECT_TRUE((m_testCart == m_hostZero));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test Zero dual */
    index += mySizesVec[0];
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[1]);
    EXPECT_TRUE((m_testCart == m_hostData));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test NnOC dual */
    index += mySizesVec[1];
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[2]);
    testNnocProjection(m_testCart);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test SOC dual */
    index += mySizesVec[2];
    m_testCart = std::vector<real_t>(m_hostCart.begin() + index, m_hostCart.begin() + index + mySizesVec[3]);
    testSocElse(m_testCart);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
}
