#include <gtest/gtest.h>
#include <numeric>
#include "../src/cones.cuh"


class ConesTest : public testing::Test {
	
    protected:
        Context context;  ///< Create one context only

        /** Prepare some host and device data */
        size_t n = 64;
        DeviceVector<real_t> d_data = DeviceVector<real_t>(n);
        std::vector<real_t> hostData = std::vector<real_t>(n);
        std::vector<real_t> hostTest = std::vector<real_t>(n);
        std::vector<real_t> hostZero = std::vector<real_t>(n);
        std::vector<real_t> hostSocA = std::vector<real_t>(n);
        std::vector<real_t> hostSocB = std::vector<real_t>(n);
        std::vector<real_t> hostSocC = std::vector<real_t>(n);
        ConesTest() {
            /** Positive and negative values in hostData */
            for (size_t i=0; i<n; i=i+2) { hostData[i] = -2. * (i + 1.); }
            for (size_t i=1; i<n; i=i+2) { hostData[i] = 2. * (i + 1.); }
            d_data.upload(hostData);
            /** Zeroes in hostZero */
            for (size_t i=1; i<n; i++) { hostZero[i] = 0.; }
            /** For testing `if` projection of SOC */
            for (size_t i=1; i<n-1; i++) { hostSocA[i] = 0.; }
            hostSocA[n-1] = 1.;
            /** For testing `else if` projection of SOC */
            for (size_t i=1; i<n-1; i++) { hostSocB[i] = 0.; }
            hostSocB[n-1] = -1.;
            /** For testing `else` projection of SOC */
            for (size_t i=1; i<n-1; i++) { hostSocC[i] = 1.; }
            hostSocC[n-1] = 0.;
        };

        virtual ~ConesTest() {}
};


TEST_F(ConesTest, RealCone) {
    Real myCone(context);
    myCone.projectOnCone(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostData));
}

TEST_F(ConesTest, RealDual) {
    Real myCone(context);
    myCone.projectOnDual(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
}

TEST_F(ConesTest, ZeroCone) {
    Zero myCone(context);
    myCone.projectOnCone(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
}

TEST_F(ConesTest, ZeroDual) {
    Zero myCone(context);
    myCone.projectOnDual(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostData));
}

TEST_F(ConesTest, NonnegativeOrthantCone) {
    NonnegativeOrthant myCone(context);
    myCone.projectOnCone(d_data);
    d_data.download(hostTest);
    for (size_t i=0; i<n; i++) { EXPECT_TRUE(hostTest[i] >= 0.); }
}

TEST_F(ConesTest, NonnegativeOrthantDual) {
    NonnegativeOrthant myCone(context);
    myCone.projectOnDual(d_data);
    d_data.download(hostTest);
    for (size_t i=0; i<n; i++) { EXPECT_TRUE(hostTest[i] >= 0.); }
}

TEST_F(ConesTest, SecondOrderConeCone) {
    SOC myCone(context);
    /** Testing `if` projection of SOC */
    d_data.upload(hostSocA);
    myCone.projectOnCone(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostSocA));
    /** Testing `else if` projection of SOC */
    d_data.upload(hostSocB);
    myCone.projectOnCone(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
    /** Testing `else` projection of SOC */
    d_data.upload(hostSocC);
    myCone.projectOnCone(d_data);
    d_data.download(hostTest);
    real_t last = hostTest[n-1];
    hostTest.pop_back();
    real_t nrm = std::sqrt(std::inner_product(hostTest.begin(), hostTest.end(), hostTest.begin(), 0.));
    EXPECT_TRUE(nrm <= last);
}

TEST_F(ConesTest, SecondOrderConeDual) {
    SOC myCone(context);
    /** Testing `if` projection of SOC */
    d_data.upload(hostSocA);
    myCone.projectOnDual(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostSocA));
    /** Testing `else if` projection of SOC */
    d_data.upload(hostSocB);
    myCone.projectOnDual(d_data);
    d_data.download(hostTest);
    EXPECT_TRUE((hostTest == hostZero));
    /** Testing `else` projection of SOC */
    d_data.upload(hostSocC);
    myCone.projectOnDual(d_data);
    d_data.download(hostTest);
    real_t last = hostTest[n-1];
    hostTest.pop_back();
    real_t nrm = std::sqrt(std::inner_product(hostTest.begin(), hostTest.end(), hostTest.begin(), 0.));
    EXPECT_TRUE(nrm <= last);
}

TEST_F(ConesTest, CartesianCone) {
    Cartesian myCone(context);
    myCone.projectOnCone(d_data);
}

TEST_F(ConesTest, CartesianDual) {
    Cartesian myCone(context);
    myCone.projectOnDual(d_data);
}
