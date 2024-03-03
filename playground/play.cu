#include "../include/stdgpu.h"
#include "wrappers.h"


int main(void) {
    std::vector<real_t> hostData;
    Context context;
    size_t n = 3;
    size_t len = n * n;
    hostData.resize(len);

    std::vector<real_t> A{1, 0, 1,
                          -1, 3, 0,
                          0, 1, 5};
    DeviceVector<real_t> d_A(A);
    DeviceVector<real_t> d_x(n);
    DeviceVector<real_t> d_b(std::vector<real_t>({1, 6, 15}));
    DeviceVector<real_t> d_workspace;
    DeviceVector<int> d_info(1);

    gpuCholeskySetup(context, n, d_workspace);
    gpuCholeskyFactor(context, n, d_workspace, d_A, d_info, true);
    gpuCholeskySolve(context, n, d_A, d_x, d_b, d_info, true);

    d_x.download(hostData);
    std::cout << "x (from device): ";
    for (size_t i = 0; i < n; i++) { std::cout << hostData[i] << " "; }
    std::cout << std::endl;

    return 0;
}