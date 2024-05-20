#ifndef PROJECTIONS_CUH
#define PROJECTIONS_CUH

#include "../include/gpu.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSoc_s1(T *, size_t, size_t, T *, T *, T *, int *, int *, T *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSoc_s2_i2(T *, size_t, size_t, int *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSoc_s2_i3(T *, size_t, size_t, T *, int *, int *, T *);


/**
 * Base projection
*/
template<typename T>
class Projection {

protected:
    size_t m_numRows = 0;
    size_t m_numCols = 0;
    size_t m_numMats = 0;

    explicit Projection(DTensor<T> &d_tensor) :
        m_numRows(d_tensor.numRows()), m_numCols(d_tensor.numCols()), m_numMats(d_tensor.numMats()) {}

    bool dimensionCheck(DTensor<T> &d_tensor) {
        if (d_tensor.numRows() != m_numRows || d_tensor.numCols() != m_numCols || d_tensor.numMats() != m_numMats) {
            std::cerr << "DTensor is [" << d_tensor.numRows() << " x " << d_tensor.numCols() << " x "
                      << d_tensor.numMats()
                      << "], but projection has dimensions [" << m_numRows << " x " << m_numCols << " x " << m_numMats
                      << "]\n";
            throw std::invalid_argument("DTensor and projection dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~Projection() {}

    virtual void project(DTensor<T> &d_tensor) = 0;
};


/**
 * Projection onto second order cones
*/
template<typename T>
class SocProjection : public Projection<T> {
private:
    size_t m_dimSOC = 0;
    size_t m_numSOCs = 0;
    std::unique_ptr<DTensor<T>> m_t_ws = nullptr;
    std::unique_ptr<DTensor<T>> m_squaredStuff_ws = nullptr;
    std::unique_ptr<DTensor<T>> m_norms = nullptr;
    std::unique_ptr<DTensor<T>> m_scalingParams = nullptr;
    std::unique_ptr<DTensor<int>> m_i2 = nullptr;
    std::unique_ptr<DTensor<int>> m_i3 = nullptr;
    size_t m_threadsPerBlock = 0;
    dim3 m_gridDims;

public:
    explicit SocProjection(DTensor<T> &d_tensor) : Projection<T>(d_tensor) {
        m_dimSOC = d_tensor.numRows();
        m_numSOCs = d_tensor.numCols();
        m_t_ws = std::make_unique<DTensor<T>>(m_numSOCs);
        m_squaredStuff_ws = std::make_unique<DTensor<T>>(m_numSOCs * (m_dimSOC - 1));
        m_norms = std::make_unique<DTensor<T>>(m_numSOCs, 1, 1, true);
        m_scalingParams = std::make_unique<DTensor<T>>(m_numSOCs, 1, 1, true);
        m_i2 = std::make_unique<DTensor<T>>(m_numSOCs, 1, 1, true);
        m_i3 = std::make_unique<DTensor<T>>(m_numSOCs, 1, 1, true);

        /* I've set the number of threads per block to 2 just for testing
         * purposes; this should be 256. We should also use DIM2BLOCKS(m_dimSOC) */
        m_threadsPerBlock = 2;
        size_t blocksDimX = (m_dimSOC / m_threadsPerBlock + (m_dimSOC % m_threadsPerBlock != 0));
        m_gridDims.x = blocksDimX;
        m_gridDims.y = m_numSOCs;

        std::cout << "Kernel dims:\n  L  THREADS per block: "
                  << m_threadsPerBlock
                  << "\n  L  Blocks: ("
                  << blocksDimX
                  << ", "
                  << m_numSOCs
                  << ")\n";
    }

    void project(DTensor<T> &d_tensor) {
        k_projectionMultiSoc_s1<<<m_gridDims, m_threadsPerBlock >>>(d_tensor.raw(), m_numSOCs, m_dimSOC, m_t_ws.raw(),
                                                                    m_squaredStuff_ws.raw(), m_norms.raw(), m_i2->raw(),
                                                                    m_i3->raw(), m_scalingParams.raw());
        k_projectionMultiSoc_s2_i2<<<m_gridDims, m_threadsPerBlock >>>(d_tensor.raw(), m_numSOCs, m_dimSOC,
                                                                       m_i2->raw());
        k_projectionMultiSoc_s2_i3<<<m_gridDims, m_threadsPerBlock >>>(d_tensor.raw(), m_numSOCs, m_dimSOC,
                                                                       m_norms.raw(), m_i2->raw(), m_i3->raw(),
                                                                       m_scalingParams.raw());
        gpuErrChk(cudaPeekAtLastError());
    }
};


#endif
