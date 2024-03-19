#include <iomanip>
#include "../include/stdgpu.h"
#include "../src/wrappers.h"


int main(void) {
    // --- gesvd only supports Nrows >= Ncols
    // --- column major memory ordering

    const int Nrows = 7;
    const int Ncols = 5;

    // --- cuSOLVE input/output parameters/arrays
    int workspaceSize = 0;
    DeviceVector<int> d_info(1);

    // --- CUDA solver initialization
    Context context;
    cusolverDnHandle_t solver_handle = context.solver();

    // --- Singular values threshold
    double threshold = 1e-12;

    // --- Setting the host, Nrows x Ncols matrix
    double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
    for(int j = 0; j < Nrows; j++)
        for(int i = 0; i < Ncols; i++)
            h_A[j + i*Nrows] = (i + j*j) * sqrt((double)(i + j));

    // --- Setting the device matrix and moving the host matrix to the device
    DeviceVector<real_t> d_A(Nrows * Ncols);

    // --- host side SVD results space
    std::vector<real_t> U(Nrows * Nrows);
    std::vector<real_t> V(Ncols * Ncols);
    std::vector<real_t> S(std::min(Nrows, Ncols));

    // --- device side SVD workspace and matrices
    DeviceVector<real_t> d_U(Nrows * Nrows);
    DeviceVector<real_t> d_V(Ncols * Ncols);
    DeviceVector<real_t> d_S(std::min(Nrows, Ncols));

    // --- CUDA SVD initialization
    DeviceVector<real_t> d_workspace;
    gpuSVDSetup(context, Nrows, Ncols, d_workspace);

    // --- CUDA SVD execution
    gpuSVDFactorise(context, Nrows, Ncols, d_workspace, d_A, d_S, d_U, d_V, d_info, true);

    // --- Moving the results from device to host
    d_S.download(S);
    d_U.download(U);
    d_V.download(V);

    for(int i = 0; i < std::min(Nrows, Ncols); i++)
        std::cout << "d_S["<<i<<"] = " << std::setprecision(15) << S[i] << std::endl;

    printf("\n\n");

    int count = 0;
    bool flag = 0;
    while (!flag) {
        if (S[count] < threshold) flag = 1;
        if (count == std::min(Nrows, Ncols)) flag = 1;
        count++;
    }
    count--;
    printf("The null space of A has dimension %i\n\n", std::min(Ncols, Nrows) - count);

    for(int j = count; j < Ncols; j++) {
        printf("Basis vector nr. %i\n", j - count);
        for(int i = 0; i < Ncols; i++)
            std::cout << "d_V["<<i<<"] = " << std::setprecision(15) << U[j*Ncols + i] << std::endl;
        printf("\n");
    }

    return 0;
}