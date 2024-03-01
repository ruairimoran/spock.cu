#include "../include/stdgpu.h"


int main(void) {
    size_t len;
    std::vector<real_t> hostData;
    Context context;
    size_t rows = 3;
    size_t cols = 2;
    len = rows * cols;
    hostData.resize(len);

    std::vector<real_t> matA1 {1, 3,
                               5, 7,
                               9, 11};
    std::vector<real_t> matB1 {0, 2,
                               4, 6,
                               8, 10};
    DeviceVector<real_t> d_matA1(matA1);
    DeviceVector<real_t> d_matB1(matB1);
    DeviceVector<real_t> d_matAPlusB(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA1, d_matB1, d_matAPlusB);
    d_matAPlusB.download(hostData);
    std::cout << "A + B (from device): ";
    for (size_t i=0; i<len; i++) { std::cout << hostData[i] << " "; }
    std::cout << std::endl;

    std::vector<real_t> matA2 {1, 5, 9,
                               3, 7, 11};
    std::vector<real_t> matB2 {0, 2,
                               4, 6,
                               8, 10};
    DeviceVector<real_t> d_matA2(matA2);
    DeviceVector<real_t> d_matB2(matB2);
    DeviceVector<real_t> d_matAtPlusB(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA2, d_matB2, d_matAtPlusB, true);
    d_matAtPlusB.download(hostData);
    std::cout << "At + B (from device): ";
    for (size_t i=0; i<len; i++) { std::cout << hostData[i] << " "; }
    std::cout << std::endl;

    std::vector<real_t> matA3 {1, 3,
                               5, 7,
                               9, 11};
    std::vector<real_t> matB3 {0, 4, 8,
                               2, 6, 10};
    DeviceVector<real_t> d_matA3(matA3);
    DeviceVector<real_t> d_matB3(matB3);
    DeviceVector<real_t> d_matAPlusBt(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA3, d_matB3, d_matAPlusBt, false, true);
    d_matAPlusBt.download(hostData);
    std::cout << "A + Bt (from device): ";
    for (size_t i=0; i<len; i++) { std::cout << hostData[i] << " "; }
    std::cout << std::endl;

    std::vector<real_t> matA4 {1, 5, 9,
                               3, 7, 11};
    std::vector<real_t> matB4 {0, 4, 8,
                               2, 6, 10};
    DeviceVector<real_t> d_matA4(matA4);
    DeviceVector<real_t> d_matB4(matB4);
    DeviceVector<real_t> d_matAtPlusBt(rows * cols);
    gpuMatAdd(context, rows, cols, d_matA4, d_matB4, d_matAtPlusBt, true, true);
    d_matAtPlusBt.download(hostData);
    std::cout << "At + Bt (from device): ";
    for (size_t i=0; i<len; i++) { std::cout << hostData[i] << " "; }
    std::cout << std::endl;
}