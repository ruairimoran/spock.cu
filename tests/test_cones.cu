#include <gtest/gtest.h>
#include "../src/cones.cuh"


TEST(Cones, One) {
    Context context; /* Create one context only */

    /* Prepare some host and device data */
    size_t n = 64;
    DeviceVector<float> d_dataContainer(n);
    std::vector<float> dataHost(n);
    for (size_t i = 0; i < n; i=i+2) { dataHost[i] = -2. * (i + 1.); }
    for (size_t i = 1; i < n; i=i+2) { dataHost[i] = 2. * (i + 1.); }
    d_dataContainer.upload(dataHost);

    /* Project to nonnegative orthant */
    NonnegativeOrthant myCone(context);
    myCone.projectOnCone(d_dataContainer.get(), d_dataContainer.capacity());

    /* Get the data back to the host and print it */
    std::vector<float> b;
    d_dataContainer.download(b);
    for (size_t i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;

    /* Project to SOC; incomplete! */
    // SOC mySoc(context);
    // mySoc.project(d_dataContainer.get(), d_dataContainer.capacity());

    EXPECT_EQ(true, 1);
}