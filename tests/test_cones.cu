#include <gtest/gtest.h>
#include "../src/cones.cuh"


class ConesTest : public testing::Test {
	
    protected:
        Context context;  ///< Create one context only

        /** Prepare some host and device data */
        size_t n = 64;
        DeviceVector<real_t> d_data;
        std::vector<real_t> hostData;
        ConesTest() {
            d_data.allocateOnDevice(n);  // why can't I allocate while declaring ?
            hostData.resize(n);
            for (size_t i=0; i<n; i=i+2) { hostData[i] = -2. * (i + 1.); }
            for (size_t i=1; i<n; i=i+2) { hostData[i] = 2. * (i + 1.); }
            d_data.upload(hostData);
        };

        virtual ~ConesTest() {}
};


TEST_F(ConesTest, RealCone) {
    Real myCone(context);
    myCone.projectOnCone(d_data);
}

TEST_F(ConesTest, RealDual) {
    Real myCone(context);
    myCone.projectOnDual(d_data);
}

TEST_F(ConesTest, ZeroCone) {
    Zero myCone(context);
    myCone.projectOnCone(d_data);
}

TEST_F(ConesTest, ZeroDual) {
    Zero myCone(context);
    myCone.projectOnDual(d_data);
}

TEST_F(ConesTest, NonnegativeOrthantCone) {
    NonnegativeOrthant myCone(context);
    myCone.projectOnCone(d_data);
}

TEST_F(ConesTest, NonnegativeOrthantDual) {
    NonnegativeOrthant myCone(context);
    myCone.projectOnDual(d_data);
}

TEST_F(ConesTest, SecondOrderConeCone) {
    SOC myCone(context);
    myCone.projectOnCone(d_data);
}

TEST_F(ConesTest, SecondOrderConeDual) {
    SOC myCone(context);
    myCone.projectOnDual(d_data);
}

TEST_F(ConesTest, CartesianCone) {
    Cartesian myCone(context);
    myCone.projectOnCone(d_data);
}

TEST_F(ConesTest, CartesianDual) {
    Cartesian myCone(context);
    myCone.projectOnDual(d_data);
}
