#include <gtest/gtest.h>
#include <numeric>
#include "cones.cuh"


class ConesTest : public testing::Test {
protected:
    ConesTest() = default;
};


TEMPLATE_WITH_TYPE_T
class ConesTestData {

public:
    /** Prepare some host and device data */
    size_t m_n = 64;
    size_t m_numConeTypes = 4;
    DTensor<T> m_d_data = DTensor<T>(m_n);
    DTensor<T> m_d_dataCart = DTensor<T>(m_n * m_numConeTypes);
    std::vector<T> m_hostData = std::vector<T>(m_n);
    std::vector<T> m_hostTest = std::vector<T>(m_n);
    std::vector<T> m_hostZero = std::vector<T>(m_n);
    std::vector<T> m_hostSocA = std::vector<T>(m_n);
    std::vector<T> m_hostSocB = std::vector<T>(m_n);
    std::vector<T> m_hostSocC = std::vector<T>(m_n);
    std::vector<T> m_hostCart = std::vector<T>(m_n * m_numConeTypes);
    std::vector<T> m_testCart;

    ConesTestData() {
        /** Positive and negative values in m_data */
        for (size_t i = 0; i < m_n; i = i + 2) { m_hostData[i] = -2. * (i + 1.); }
        for (size_t i = 1; i < m_n; i = i + 2) { m_hostData[i] = 2. * (i + 1.); }
        m_d_data.upload(m_hostData);  ///< Main vector for projection testing
        /** Zeroes in m_zero */
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
    };

    virtual ~ConesTestData() = default;
};

TEMPLATE_WITH_TYPE_T
void testNnocProjection(std::vector<T> testVec) {
    for (size_t i = 0; i < testVec.size(); i++) { EXPECT_TRUE(testVec[i] >= 0.); }
}

TEMPLATE_WITH_TYPE_T
void testSocElse(std::vector<T> testVec, T epsilon) {
    T last = testVec[testVec.size() - 1];
    testVec.pop_back();
    T nrm = std::sqrt(std::inner_product(testVec.begin(), testVec.end(), testVec.begin(), 0.));
    EXPECT_TRUE(nrm <= last + epsilon);
}

TEMPLATE_WITH_TYPE_T
void testUniverseCone(ConesTestData<T> &d) {
    UniverseCone<T> myCone(d.m_n);
    myCone.project(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostData));
}

TEST_F(ConesTest, UniverseCone) {
    ConesTestData<float> df;
    testUniverseCone<float>(df);
    ConesTestData<double> dd;
    testUniverseCone<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testUniverseDual(ConesTestData<T> &d) {
    UniverseCone<T> myCone(d.m_n);
    myCone.projectOnDual(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostZero));
}

TEST_F(ConesTest, universeDual) {
    ConesTestData<float> df;
    testUniverseDual<float>(df);
    ConesTestData<double> dd;
    testUniverseDual<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testZeroCone(ConesTestData<T> &d) {
    ZeroCone<T> myCone(d.m_n);
    myCone.project(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostZero));
}

TEST_F(ConesTest, zeroCone) {
    ConesTestData<float> df;
    testZeroCone<float>(df);
    ConesTestData<double> dd;
    testZeroCone<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testZeroDual(ConesTestData<T> &d) {
    ZeroCone<T> myCone(d.m_n);
    myCone.projectOnDual(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostData));
}

TEST_F(ConesTest, zeroDual) {
    ConesTestData<float> df;
    testZeroDual<float>(df);
    ConesTestData<double> dd;
    testZeroDual<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testNonnegativeOrthantCone(ConesTestData<T> &d) {
    NonnegativeOrthantCone<T> myCone(d.m_n);
    myCone.project(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    testNnocProjection(d.m_hostTest);
}

TEST_F(ConesTest, nonnegativeOrthantCone) {
    ConesTestData<float> df;
    testNonnegativeOrthantCone<float>(df);
    ConesTestData<double> dd;
    testNonnegativeOrthantCone<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testNonnegativeOrthantDual(ConesTestData<T> &d) {
    NonnegativeOrthantCone<T> myCone(d.m_n);
    myCone.projectOnDual(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    testNnocProjection(d.m_hostTest);
}

TEST_F(ConesTest, nonnegativeOrthantDual) {
    ConesTestData<float> df;
    testNonnegativeOrthantDual<float>(df);
    ConesTestData<double> dd;
    testNonnegativeOrthantDual<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testSecondOrderConeCone(ConesTestData<T> &d, T epsilon) {
    SecondOrderCone<T> myCone(d.m_n);
    /** Testing `if` projection of SOC */
    d.m_d_data.upload(d.m_hostSocA);
    myCone.project(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostSocA));
    /** Testing `else if` projection of SOC */
    d.m_d_data.upload(d.m_hostSocB);
    myCone.project(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostZero));
    /** Testing `else` projection of SOC */
    d.m_d_data.upload(d.m_hostSocC);
    myCone.project(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    testSocElse(d.m_hostTest, epsilon);
}

TEST_F(ConesTest, secondOrderConeCone) {
    ConesTestData<float> df;
    testSecondOrderConeCone<float>(df, TEST_PRECISION_LOW);
    ConesTestData<double> dd;
    testSecondOrderConeCone<double>(dd, TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void testSecondOrderConeDual(ConesTestData<T> &d, T epsilon) {
    SecondOrderCone<T> myCone(d.m_n);
    /** Testing `if` projection of SOC */
    d.m_d_data.upload(d.m_hostSocA);
    myCone.projectOnDual(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostSocA));
    /** Testing `else if` projection of SOC */
    d.m_d_data.upload(d.m_hostSocB);
    myCone.projectOnDual(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    EXPECT_TRUE((d.m_hostTest == d.m_hostZero));
    /** Testing `else` projection of SOC */
    d.m_d_data.upload(d.m_hostSocC);
    myCone.projectOnDual(d.m_d_data);
    d.m_d_data.download(d.m_hostTest);
    testSocElse(d.m_hostTest, epsilon);
}

TEST_F(ConesTest, secondOrderConeDual) {
    ConesTestData<float> df;
    testSecondOrderConeDual<float>(df, TEST_PRECISION_LOW);
    ConesTestData<double> dd;
    testSecondOrderConeDual<double>(dd, TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void testCartesianCone(ConesTestData<T> &d, T epsilon) {
    // for (size_t i=0; i<m_n*m_numConeTypes; i++) { std::cerr << m_dataCart[i] << " "; }  ///< For debugging
    d.m_d_dataCart.upload(d.m_hostCart);
    UniverseCone<T> myUniv(d.m_n);
    ZeroCone<T> myZero(d.m_n);
    NonnegativeOrthantCone<T> myNnoc(d.m_n);
    SecondOrderCone<T> mySoc(d.m_n);
    Cartesian<T> myCone;
    myCone.addCone(myUniv);
    myCone.addCone(myZero);
    myCone.addCone(myNnoc);
    myCone.addCone(mySoc);
    myCone.project(d.m_d_dataCart);
    d.m_d_dataCart.download(d.m_hostCart);
    /** Test Universe cone */
    size_t index = 0;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    EXPECT_TRUE((d.m_testCart == d.m_hostData));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test Zero cone */
    index += d.m_n;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    EXPECT_TRUE((d.m_testCart == d.m_hostZero));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test NnOC cone */
    index += d.m_n;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    testNnocProjection(d.m_testCart);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test SOC cone */
    index += d.m_n;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    testSocElse(d.m_testCart, epsilon);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
}

TEST_F(ConesTest, cartesianCone) {
    ConesTestData<float> df;
    testCartesianCone<float>(df, TEST_PRECISION_LOW);
    ConesTestData<double> dd;
    testCartesianCone<double>(dd, TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void testCartesianDual(ConesTestData<T> &d, T epsilon) {
    // for (size_t i=0; i<m_n*m_numConeTypes; i++) { std::cerr << m_dataCart[i] << " "; }  ///< For debugging
    d.m_d_dataCart.upload(d.m_hostCart);
    UniverseCone<T> myUniv(d.m_n);
    ZeroCone<T> myZero(d.m_n);
    NonnegativeOrthantCone<T> myNnoc(d.m_n);
    SecondOrderCone<T> mySoc(d.m_n);
    Cartesian<T> myCone;
    myCone.addCone(myUniv);
    myCone.addCone(myZero);
    myCone.addCone(myNnoc);
    myCone.addCone(mySoc);
    myCone.projectOnDual(d.m_d_dataCart);
    d.m_d_dataCart.download(d.m_hostCart);
    /** Test Universe dual */
    size_t index = 0;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    EXPECT_TRUE((d.m_testCart == d.m_hostZero));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test Zero dual */
    index += d.m_n;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    EXPECT_TRUE((d.m_testCart == d.m_hostData));
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test NnOC dual */
    index += d.m_n;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    testNnocProjection(d.m_testCart);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
    /** Test SOC dual */
    index += d.m_n;
    d.m_testCart = std::vector<T>(d.m_hostCart.begin() + index, d.m_hostCart.begin() + index + d.m_n);
    testSocElse(d.m_testCart, epsilon);
    // for (size_t i=0; i<m_n; i++) { std::cerr << m_testCart[i] << " "; }  ///< For debugging
}

TEST_F(ConesTest, cartesianDual) {
    ConesTestData<float> df;
    testCartesianDual<float>(df, TEST_PRECISION_LOW);
    ConesTestData<double> dd;
    testCartesianDual<double>(dd, TEST_PRECISION_HIGH);
}

TEMPLATE_WITH_TYPE_T
void testDimension(ConesTestData<T> &d) {
    UniverseCone<T> myUniv(d.m_n);
    EXPECT_EQ(myUniv.dimension(), d.m_n);
}

TEST_F(ConesTest, dimension) {
    ConesTestData<float> df;
    testDimension<float>(df);
    ConesTestData<double> dd;
    testDimension<double>(dd);
}

TEMPLATE_WITH_TYPE_T
void testFailDimension(ConesTestData<T> &d) {
    d.m_d_data.upload(d.m_hostData);
    UniverseCone<T> myCone(d.m_n + 1);
    EXPECT_ANY_THROW(myCone.project(d.m_d_data));
}

TEST_F(ConesTest, failDimension) {
    ConesTestData<float> df;
    testFailDimension<float>(df);
    ConesTestData<double> dd;
    testFailDimension<double>(dd);
}
