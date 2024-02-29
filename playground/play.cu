#include "../include/stdgpu.h"


int main(void) {
    size_t len;
    Context context;
    size_t numStates = 3;
    size_t numInputs = 2;
    std::vector<real_t> matA {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<real_t> matB {1, 2, 3, 4, 5, 6};
    DeviceVector<real_t> d_matA(matA);
    DeviceVector<real_t> d_matB(matB);
    DeviceVector<real_t> d_matAB(numStates * numInputs);
    DeviceVector<real_t> d_matAtB(numStates * numInputs);
    std::vector<real_t> hostData;

    gpuMatMul(context, numStates, numStates, numInputs, d_matA, d_matB, d_matAB);

    len = numStates * numInputs;
    hostData.resize(len);
    d_matAB.download(hostData);
    std::cout << "AB (from device): ";
    for (size_t i=0; i<len; i++) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    gpuMatMul(context, numStates, numStates, numInputs, d_matA, d_matB, d_matAtB, true);

    len = numStates * numInputs;
    hostData.resize(len);
    d_matAtB.download(hostData);
    std::cout << "AtB (from device): ";
    for (size_t i=0; i<len; i++) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;
}