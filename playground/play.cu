#include <tensor.cuh>
#include <vector>
#include <iostream>

#define real_t double

__global__ void k_projectionMultiSoc_s1(real_t *data,
                                        size_t numCones,
                                        size_t coneDimension,
                                        real_t *t_ws,
                                        real_t *squaredElements_ws,
                                        real_t *norms,
                                        int *i2,
                                        int *i3,
                                        real_t *scaling) {
    /* Copy data to workspace */
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idSoc = blockIdx.y;
    const unsigned int s = idSoc * coneDimension + tid;
    if (idSoc < numCones) t_ws[idSoc] = data[coneDimension * (idSoc + 1) - 1];
    if (idSoc < numCones && tid < coneDimension - 1) {
        real_t temp = data[s];
        squaredElements_ws[s - idSoc] = temp * temp;
    }
    __syncthreads(); /* sync threads in each block */

    /* Since each block corresponds to a different SOC and the dimension of
     * each SOC is not going to be too large in general#, the addition will
     * be performed by using atomicAdd. In order for the synchronisation to
     * be block-wise (and not device-wide), we will use shared memory.
     * # for this reason we won't do any map-reduce-type summation
     */
    extern __shared__ real_t sharedMem[];
    if (idSoc < numCones && tid < coneDimension - 1)
        sharedMem[tid] = squaredElements_ws[idSoc * (coneDimension - 1) + tid];
    __syncthreads();

    /* and now do the addition atomically */
    if (idSoc < numCones && tid < coneDimension - 1) {
        atomicAdd(&norms[idSoc], sharedMem[tid]);
    }
    __syncthreads();

    /* Final touch: apply the square root to determine the Euclidean norms */
    if (idSoc < numCones && tid == 0) {
        norms[idSoc] = sqrt(norms[idSoc]);
    }
    __syncthreads();

    /* populate sets i2 and i3 and compute scaling parameters */
    if (idSoc < numCones && tid == 0) {
        real_t nrm_j = norms[idSoc];
        real_t t_j = t_ws[idSoc];
        i2[idSoc] = nrm_j <= -t_j;
        i3[idSoc] = nrm_j > t_j && nrm_j > -t_j;  // should this not be < >
        scaling[idSoc] = (nrm_j + t_j) / (2 * nrm_j);
    }
}

__global__ void k_projectionMultiSoc_s2_i2(real_t *data,
                                           size_t numCones,
                                           size_t coneDimension,
                                           int *i2) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idSoc = blockIdx.y;
    const unsigned int s = idSoc * coneDimension + tid;
    if (idSoc < numCones && tid < coneDimension) data[s] *= 1 - i2[idSoc];
}

__global__ void k_projectionMultiSoc_s2_i3(real_t *data,
                                           size_t numCones,
                                           size_t coneDimension,
                                           real_t *norms,
                                           int *i2,
                                           int *i3,
                                           real_t *scaling) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idSoc = blockIdx.y;
    const unsigned int s = idSoc * coneDimension + tid;

    if (idSoc < numCones && tid < coneDimension - 1) {
        real_t multiplier = i3[idSoc] * scaling[idSoc] + 1 - i3[idSoc];
        data[s] *= multiplier;
    }
    if (idSoc < numCones && tid == coneDimension - 1) {
        int c_i2 = i2[idSoc];
        int c_i3 = i3[idSoc];
        int c_i1 = (1 - c_i2) * (1 - c_i3);
        data[s] = c_i1 * data[s] + (1 - c_i1) * (1 - c_i2) * (c_i3 * (scaling[idSoc] * norms[idSoc] - 1) + 1);
    }
}


void projectSocBatched(DTensor<real_t> &x) {
    size_t dimSOC = x.numRows();
    size_t numSOCs = x.numCols();

    /* Allocate workspace memory
     * (there will become class attributes later) */
    DTensor<real_t> t_ws(numSOCs);
    DTensor<real_t> squaredStuff_ws(numSOCs * (dimSOC - 1));
    DTensor<real_t> norms(numSOCs, 1, 1, true);
    DTensor<real_t> scalingParams(numSOCs, 1, 1, true);
    DTensor<int> i2(numSOCs, 1, 1, true);
    DTensor<int> i3(numSOCs, 1, 1, true);

    /* I've set the number of threads per block to 2 just for testing
     * purposes; this should be 256. We should also use DIM2BLOCKS(m_dimSOC) */
    constexpr size_t threadsPerBlock = 2;
    size_t blocksDimX = (dimSOC / threadsPerBlock + (dimSOC % threadsPerBlock != 0));
    dim3 gridDims(blocksDimX, numSOCs);

    std::cout << "Kernel dims:\n  L  THREADS per block: "
              << threadsPerBlock
              << "\n  L  Blocks: ("
              << blocksDimX
              << ", "
              << numSOCs
              << ")\n";

    k_projectionMultiSoc_s1<<<gridDims, threadsPerBlock >>>(x.raw(), numSOCs, dimSOC, t_ws.raw(),
                                                            squaredStuff_ws.raw(), norms.raw(), i2.raw(), i3.raw(),
                                                            scalingParams.raw());
    k_projectionMultiSoc_s2_i2<<<gridDims, threadsPerBlock >>>(x.raw(), numSOCs, dimSOC, i2.raw());
    k_projectionMultiSoc_s2_i3<<<gridDims, threadsPerBlock >>>(x.raw(), numSOCs, dimSOC, norms.raw(), i2.raw(),
                                                               i3.raw(), scalingParams.raw());
    gpuErrChk(cudaPeekAtLastError());
}

int main() {
    std::vector<real_t> dat = {1., 2., 3., 4., 0.5,
                               5., 6., 7., 8., -200,
                               9., -10., 11., -12., 100};
    DTensor<real_t> x(dat, 5, 3);
    std::cout << "x (before) : " << x;
    projectSocBatched(x); /* project in-place */
    std::cout << "x (after) : " << x;

    gpuErrChk(cudaDeviceReset());
    return 0;
}