#ifndef RISKS_CUH
#define RISKS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


//__global__ void d_avarVecAddB(real_t *vec, size_t node, size_t *numCh, size_t *chFrom, real_t *probs);


/**
 * Base class for a coherent risk
 * that can be described by the tuple
 * (E, F, K, b) = (matrix, matrix, cone, vector)
 */
template<typename T>
class CoherentRisk {

protected:
    size_t m_nodeIndex = 0;
    size_t m_dimension = 0;  ///< Number of children
    std::unique_ptr<DTensor<T>> m_d_E = nullptr;
    std::unique_ptr<DTensor<T>> m_d_F = nullptr;
    std::unique_ptr<Cartesian> m_K = nullptr;
    std::unique_ptr<DTensor<T>> m_d_b = nullptr;

    explicit CoherentRisk(size_t node, size_t numChildren) : m_nodeIndex(node), m_dimension(numChildren) {}

    bool dimensionCheck(DTensor<T> &d_vec) {
        if (d_vec.numRows() != m_dimension || d_vec.numCols() != 1 || d_vec.numMats() != 1) {
            std::cerr << "DTensor is [" << d_vec.numRows() << " x " << d_vec.numCols() << " x " << d_vec.numMats()
                      << "], but risk has dimensions [" << m_dimension << " x " << 1 << " x " << 1 << "]\n";
            throw std::invalid_argument("DTensor and risk dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~CoherentRisk() {}

    virtual Cartesian &cone() { return *m_K; }

    virtual void print() = 0;
};


/**
 * Average Value at Risk (AVaR)
 * - parameters: alpha
 * - matrix F is not needed (nullptr)
*/
template<typename T>
class AVaR : public CoherentRisk<T> {

protected:
    T m_alpha = 0;
    DTensor<size_t> &m_d_chFrom;
    DTensor<size_t> &m_d_chTo;
    DTensor<T> &m_d_condProbs;
    NonnegativeOrthantCone m_nnoc;
    ZeroCone m_zero;

public:
    explicit AVaR(T alpha, size_t node, size_t numChildren, DTensor<size_t> &d_childFrom, DTensor<size_t> &d_childTo,
                  DTensor<T> &d_condProbs) : CoherentRisk<T>(node, numChildren),
                                             m_alpha(alpha),
                                             m_d_chFrom(d_childFrom),
                                             m_d_chTo(d_childTo),
                                             m_d_condProbs(d_condProbs),
                                             m_nnoc(numChildren * 2),
                                             m_zero(1) {
        CoherentRisk<T>::m_K = std::make_unique<Cartesian>();
        CoherentRisk<T>::m_K->addCone(m_nnoc);
        CoherentRisk<T>::m_K->addCone(m_zero);
    }

    void print() {
        std::cout << "Node: " << CoherentRisk<T>::m_nodeIndex << ", Risk: AVaR, \n";
        std::cout << "K: ";
        if (CoherentRisk<T>::m_K) {
            CoherentRisk<T>::m_K->print();
        } else {
            std::cout << "NO CONE TO PRINT.\n";
        }
    }

};


#endif
