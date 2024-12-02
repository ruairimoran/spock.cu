#ifndef RISKS_CUH
#define RISKS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectionMultiIndexedNnoc(T *, size_t, int *, int *);


/**
 * Base class for a coherent risk
 * that can be described by the tuple
 * (E, F, K, b) = (matrix, matrix, cone, vector)
 */
template<typename T>
class CoherentRisk {

protected:
    size_t m_dim;
    std::unique_ptr<DTensor<T>> m_d_nullspaceProjectionMatrix = nullptr;
    std::unique_ptr<DTensor<T>> m_d_b = nullptr;
    std::unique_ptr<DTensor<T>> m_d_bTr = nullptr;

    explicit CoherentRisk(std::string path) {
        m_d_nullspaceProjectionMatrix = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromTextFile(path + "NNtr", rowMajor));
        m_d_b = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromTextFile(path + "b", rowMajor));
        m_d_bTr = std::make_unique<DTensor<T>>(m_d_b->tr());
        m_dim = m_d_b->numEl();
    }

    bool dimensionCheck(DTensor<T> &d_tensor) {
        if (d_tensor.numEl() != m_dim) {
            err << "[Risk] Given DTensor size (" << d_tensor.numEl()
                << "), but projection setup for (" << m_dim << "]\n";
            throw std::invalid_argument(err.str());
        }
        return true;
    }

    virtual std::ostream &print(std::ostream &out) const { return out; };

public:
    virtual ~CoherentRisk() = default;

    virtual size_t dimension() { return m_dim; }

    virtual DTensor<T> &nullspaceProj() { return *m_d_nullspaceProjectionMatrix; }

    virtual DTensor<T> &b() { return *m_d_b; }

    virtual DTensor<T> &bTr() { return *m_d_bTr; }

    virtual void projectDual(DTensor<T> &) {};

    friend std::ostream &operator<<(std::ostream &out, const CoherentRisk<T> &data) { return data.print(out); }
};


/**
 * Average Value at Risk (AVaR)
*/
template<typename T>
class AVaR : public CoherentRisk<T> {

protected:
    std::vector<int> m_idxNnoc;
    std::unique_ptr<DTensor<int>> m_d_idxNnoc = nullptr;
    std::unique_ptr<DTensor<int>> m_d_idx = nullptr;
    std::unique_ptr<DTensor<int>> m_d_zeros = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Risk: AVaR, \n";
        return out;
    }

public:
    explicit AVaR(std::string path, std::vector<size_t> &numCh) : CoherentRisk<T>(path) {
        /* The zero cone is extended for nodes with fewer children.
         * As we only project on the dual, some elements will never be changed.
         */
        size_t riskDim = this->b().numRows();
        for (size_t i = 0; i < this->b().numMats(); i++) {
            size_t nnocDim = numCh[i] * 2;
            for (size_t j = 0; j < riskDim; j++) {
                if (j < nnocDim) m_idxNnoc.push_back(1);
                else m_idxNnoc.push_back(0);
            }
        }
        size_t n = this->b().numEl();
        m_d_idxNnoc = std::make_unique<DTensor<int>>(m_idxNnoc, n);
        m_d_idx = std::make_unique<DTensor<int>>(n);
        m_d_zeros = std::make_unique<DTensor<int>>(n, 1, 1, true);
    }

    void projectDual(DTensor<T> &d_tensor) {
        this->dimensionCheck(d_tensor);
        m_d_zeros->deviceCopyTo(*m_d_idx);
        k_projectionMultiIndexedNnoc<<<numBlocks(this->m_dim, TPB), TPB>>>(d_tensor.raw(),
                                                                           this->m_dim,
                                                                           m_d_idxNnoc->raw(),
                                                                           m_d_idx->raw());
    }
};


#endif
