#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include "../src/cones.cuh"


class ConesTest : public testing::Test {
	
    protected:
        Context context;  ///< Create one context only

        /** Prepare some host and device data */
        size_t n = 64;
        size_t numConeTypes = 4;
        DeviceVector<real_t> d_data = DeviceVector<real_t>(n);
        std::vector<real_t> hostData = std::vector<real_t>(n);
        std::vector<real_t> hostTest = std::vector<real_t>(n);
        std::vector<real_t> hostZero = std::vector<real_t>(n);
        std::vector<real_t> hostSocA = std::vector<real_t>(n);
        std::vector<real_t> hostSocB = std::vector<real_t>(n);
        std::vector<real_t> hostSocC = std::vector<real_t>(n);
        std::vector<real_t> hostCart = std::vector<real_t>(n * numConeTypes);
        std::vector<real_t> testCart;
        ConesTest() {
            /** Positive and negative values in hostData */
            for (size_t i=0; i<n; i=i+2) { hostData[i] = -2. * (i + 1.); }
            for (size_t i=1; i<n; i=i+2) { hostData[i] = 2. * (i + 1.); }
            d_data.upload(hostData);  ///< Main vector for projection testing
            /** Zeroes in hostZero */
            for (size_t i=0; i<n; i++) { hostZero[i] = 0.; }
            /** For testing `if` projection of SOC */
            for (size_t i=0; i<n-1; i++) { hostSocA[i] = 0.; }
            hostSocA[n-1] = 1.;
            /** For testing `else if` projection of SOC */
            for (size_t i=0; i<n-1; i++) { hostSocB[i] = 0.; }
            hostSocB[n-1] = -1.;
            /** For testing `else` projection of SOC */
            for (size_t i=0; i<n-1; i++) { hostSocC[i] = 1.; }
            hostSocC[n-1] = 0.;
            /** For testing Cartesian cone of all types */
            for (size_t i=0; i<n*numConeTypes; i++) {
                if (n * 0 <= i && i < n * 1) hostCart[i] = hostData[i%n];  ///< For projecting Cartesian::Real
                if (n * 1 <= i && i < n * 2) hostCart[i] = hostData[i%n];  ///< For projecting Cartesian::Zero
                if (n * 2 <= i && i < n * 3) hostCart[i] = hostData[i%n];  ///< For projecting Cartesian::NnOC
                if (n * 3 <= i && i < n * 4) hostCart[i] = hostSocC[i%n];  ///< For projecting Cartesian::SOC
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


TEST_F(ConesTest, RealCone) {
    Real myCone(context);
    myCone.projectOnCone(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostData));
}

TEST_F(ConesTest, RealDual) {
    Real myCone(context);
    myCone.projectOnDual(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
}

TEST_F(ConesTest, ZeroCone) {
    Zero myCone(context);
    myCone.projectOnCone(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
}

TEST_F(ConesTest, ZeroDual) {
    Zero myCone(context);
    myCone.projectOnDual(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostData));
}

TEST_F(ConesTest, NonnegativeOrthantCone) {
    NnOC myCone(context);
    myCone.projectOnCone(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    testNnocProjection(hostTest);
}

TEST_F(ConesTest, NonnegativeOrthantDual) {
    NnOC myCone(context);
    myCone.projectOnDual(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    testNnocProjection(hostTest);
}

TEST_F(ConesTest, SecondOrderConeCone) {
    SOC myCone(context);
    /** Testing `if` projection of SOC */
    d_data.upload(hostSocA);
    myCone.projectOnCone(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostSocA));
    /** Testing `else if` projection of SOC */
    d_data.upload(hostSocB);
    myCone.projectOnCone(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
    /** Testing `else` projection of SOC */
    d_data.upload(hostSocC);
    myCone.projectOnCone(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    testSocElse(hostTest);
}

TEST_F(ConesTest, SecondOrderConeDual) {
    SOC myCone(context);
    /** Testing `if` projection of SOC */
    d_data.upload(hostSocA);
    myCone.projectOnDual(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostSocA));
    /** Testing `else if` projection of SOC */
    d_data.upload(hostSocB);
    myCone.projectOnDual(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
    /** Testing `else` projection of SOC */
    d_data.upload(hostSocC);
    myCone.projectOnDual(d_data.get(), d_data.capacity());
    d_data.download(hostTest);
    testSocElse(hostTest);
}

TEST_F(ConesTest, CartesianCone) {
    // for (size_t i=0; i<n*numConeTypes; i++) { std::cerr << hostCart[i] << " "; }  ///< For debugging
    d_data.upload(hostCart);
    Real myRealCone(context);
    Zero myZeroCone(context);
    NnOC myNnocCone(context);
    SOC mySocCone(context);
    std::vector<ConvexCone*> myConesVec {&myRealCone, &myZeroCone, &myNnocCone, &mySocCone};
    std::vector<size_t> mySizesVec(numConeTypes, n);
    Cartesian myCone(context, myConesVec, mySizesVec);
    myCone.projectOnCone(d_data.get());
    d_data.download(hostCart);
    /** Test Real cone */
    size_t index = 0;
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[0]);
    EXPECT_TRUE((testCart == hostData));
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
    /** Test Zero cone */
    index += mySizesVec[0];
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[1]);
    EXPECT_TRUE((testCart == hostZero));
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
    /** Test NnOC cone */
    index += mySizesVec[1];
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[2]);
    testNnocProjection(testCart);
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
    /** Test SOC cone */
    index += mySizesVec[2];
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[3]);
    testSocElse(testCart);
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
}

TEST_F(ConesTest, CartesianDual) {
    // for (size_t i=0; i<n*numConeTypes; i++) { std::cerr << hostCart[i] << " "; }  ///< For debugging
    d_data.upload(hostCart);
    Real myRealCone(context);
    Zero myZeroCone(context);
    NnOC myNnocCone(context);
    SOC mySocCone(context);
    std::vector<ConvexCone*> myConesVec {&myRealCone, &myZeroCone, &myNnocCone, &mySocCone};
    std::vector<size_t> mySizesVec(numConeTypes, n);
    Cartesian myCone(context, myConesVec, mySizesVec);
    myCone.projectOnDual(d_data.get());
    d_data.download(hostCart);
    /** Test Real dual */
    size_t index = 0;
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[0]);
    EXPECT_TRUE((testCart == hostZero));
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
    /** Test Zero dual */
    index += mySizesVec[0];
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[1]);
    EXPECT_TRUE((testCart == hostData));
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
    /** Test NnOC dual */
    index += mySizesVec[1];
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[2]);
    testNnocProjection(testCart);
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
    /** Test SOC dual */
    index += mySizesVec[2];
    testCart = std::vector<real_t>(hostCart.begin() + index, hostCart.begin() + index + mySizesVec[3]);
    testSocElse(testCart);
    // for (size_t i=0; i<n; i++) { std::cerr << testCart[i] << " "; }  ///< For debugging
}
