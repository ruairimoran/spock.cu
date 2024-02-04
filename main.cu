#include "src/cones.cuh"
#include <iostream>


int main() {
    Context context; /* Create one context only */

    /* Prepare some host and device data */
    size_t n = 8;
    DeviceVector<real_t> d_dataContainer(n);
    std::vector<real_t> dataHost(n);
    for (size_t i = 0; i < n; i=i+2) { dataHost[i] = -2. * (i + 1.); }
    for (size_t i = 1; i < n; i=i+2) { dataHost[i] = 2. * (i + 1.); }
    d_dataContainer.upload(dataHost);

    /* Project to nonnegative orthant */
    NonnegativeOrthant myCone(context);
    myCone.projectOnCone(d_dataContainer.get(), d_dataContainer.capacity());

    /* Get the data back to the host and print it */
    std::vector<real_t> b;
    d_dataContainer.download(b);
    for (size_t i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;

    /* Project to SOC; incomplete! */
    SOC mySoc(context);
    mySoc.projectOnCone(d_dataContainer.get(), d_dataContainer.capacity());
}
