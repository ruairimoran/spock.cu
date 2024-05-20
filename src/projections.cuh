#ifndef PROJECTIONS_CUH
#define PROJECTIONS_CUH

#include "../include/gpu.cuh"


/**
 * Base projection
*/
template<typename T>
class Projection {

protected:
    size_t m_dimension = 0;

    explicit Projection(size_t node, size_t dim) : m_dimension(dim) {}

    bool dimensionCheck(DTensor<T> &d_vec) {
        if (d_vec.numRows() != m_dimension || d_vec.numCols() != 1 || d_vec.numMats() != 1) {
            std::cerr << "DTensor is [" << d_vec.numRows() << " x " << d_vec.numCols() << " x " << d_vec.numMats()
                      << "], but projection has dimensions [" << m_dimension << " x " << 1 << " x " << 1 << "]\n";
            throw std::invalid_argument("DTensor and projection dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~Projection() {}

    size_t dimension() { return m_dimension; }

    virtual void project(DTensor<T> &d_vectors) = 0;
};


/**
 * Projection onto second order cones
*/
template<typename T>
class SocProjection : public Projection<T> {

public:
    explicit SocProjection(size_t node, size_t dim) : Projection<T>(node, dim) {}

    void project(DTensor<real_t> &d_vecs) {
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
         * purposes; this should be 256. We should also use DIM2BLOCKS(dimSOC) */
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
};

#endif PROJECTIONS_CUH
