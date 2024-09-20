#ifndef PROJECTIONS_CUH
#define PROJECTIONS_CUH

#include "../include/gpu.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep1(T *, size_t, size_t, T *, T *, T *, int *, int *, T *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep2(T *, size_t, size_t, int *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiSocStep3(T *, size_t, size_t, T *, int *, int *, T *);


/**
 * Base projection for parallel implementations of projections
*/
TEMPLATE_WITH_TYPE_T
class Projectable {

protected:
    size_t m_numRows = 0;
    size_t m_numCols = 0;
    size_t m_numMats = 0;


    explicit Projectable(DTensor<T> &d_tensor) :
        m_numRows(d_tensor.numRows()), m_numCols(d_tensor.numCols()), m_numMats(d_tensor.numMats()) {}

    bool dimensionCheck(DTensor<T> &d_tensor) {
        if (d_tensor.numRows() != m_numRows || d_tensor.numCols() != m_numCols || d_tensor.numMats() != m_numMats) {
            std::cerr << "Given DTensor [" << d_tensor.numRows() << " x " << d_tensor.numCols() << " x "
                      << d_tensor.numMats()
                      << "], but projection setup for [" << m_numRows << " x " << m_numCols << " x " << m_numMats
                      << "]\n";
            throw std::invalid_argument("DTensor and projection dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~Projectable() {}

    virtual void project(DTensor<T> &d_tensor) = 0;
};


/**
 * Projection onto second order cones
*/
TEMPLATE_WITH_TYPE_T
class SocProjection : public Projectable<T> {
private:
    std::unique_ptr<DTensor<T>> m_d_lastElementOfSocs = nullptr;
    std::unique_ptr<DTensor<T>> m_d_squaredElements = nullptr;
    std::unique_ptr<DTensor<T>> m_d_norms = nullptr;
    std::unique_ptr<DTensor<T>> m_d_scalingParams = nullptr;
    std::unique_ptr<DTensor<int>> m_d_i2 = nullptr;
    std::unique_ptr<DTensor<int>> m_d_i3 = nullptr;
    std::unique_ptr<DTensor<T>> m_d_zeros = nullptr;
    size_t m_threadsPerBlock = 0;
    dim3 m_gridDims;
    size_t m_sharedMemBytes = 0;

public:
    explicit SocProjection(DTensor<T> &d_tensor) : Projectable<T>(d_tensor) {
        m_d_zeros = std::make_unique<DTensor<T>>(this->m_numCols, 1, 1, true);
        m_d_lastElementOfSocs = std::make_unique<DTensor<T>>(this->m_numCols);
        m_d_squaredElements = std::make_unique<DTensor<T>>(this->m_numCols * (this->m_numRows - 1));
        m_d_norms = std::make_unique<DTensor<T>>(this->m_numCols);
        m_d_scalingParams = std::make_unique<DTensor<T>>(this->m_numCols, 1, 1, true);
        m_d_i2 = std::make_unique<DTensor<int>>(this->m_numCols, 1, 1, true);
        m_d_i3 = std::make_unique<DTensor<int>>(this->m_numCols, 1, 1, true);
        m_threadsPerBlock = TPB;
        size_t blocksDimX = numBlocks(this->m_numRows, m_threadsPerBlock);
        m_gridDims.x = blocksDimX;
        m_gridDims.y = this->m_numCols;
        size_t sharedMemMultiplier = (this->m_numRows > TPB) ? TPB : this->m_numRows;
        m_sharedMemBytes = sizeof(T) * sharedMemMultiplier;
    }

    void project(DTensor<T> &d_tensor) {
        this->dimensionCheck(d_tensor);
        m_d_zeros->deviceCopyTo(*m_d_norms);
        k_projectionMultiSocStep1<<<m_gridDims, m_threadsPerBlock, m_sharedMemBytes>>>(d_tensor.raw(), this->m_numCols,
                                                                                       this->m_numRows,
                                                                                       m_d_lastElementOfSocs->raw(),
                                                                                       m_d_squaredElements->raw(),
                                                                                       m_d_norms->raw(), m_d_i2->raw(),
                                                                                       m_d_i3->raw(),
                                                                                       m_d_scalingParams->raw());
        k_projectionMultiSocStep2<<<m_gridDims, m_threadsPerBlock>>>(d_tensor.raw(), this->m_numCols, this->m_numRows,
                                                                     m_d_i2->raw());
        k_projectionMultiSocStep3<<<m_gridDims, m_threadsPerBlock>>>(d_tensor.raw(), this->m_numCols, this->m_numRows,
                                                                     m_d_norms->raw(), m_d_i2->raw(), m_d_i3->raw(),
                                                                     m_d_scalingParams->raw());
    }
};


#endif
