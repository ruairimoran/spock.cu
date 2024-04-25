#ifndef RISKS_CUH
#define RISKS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


__global__ void d_avarVecAddB(real_t *vec, size_t node, size_t *numCh, size_t *chFrom, real_t *probs);


/**
 * Base class for a coherent risk
 * that can be described by the tuple
 * (E, F, K, b)
 */
class CoherentRisk {

protected:
    size_t m_node = 0;
    DTensor<size_t> &m_d_numCh;
    size_t m_numCh = 0;
    size_t m_dimension = 0;

    explicit CoherentRisk(size_t node, DTensor<size_t> &numCh) :
        m_node(node),
        m_d_numCh(numCh),
        m_numCh(m_d_numCh(m_node)),
        m_dimension(m_numCh * 2 + 1) {}

    bool dimension_check(DTensor<real_t> &d_vec) {
        if (d_vec.numRows() != m_dimension || d_vec.numCols() != 1 || d_vec.numMats() != 1) {
            std::cerr << "DTensor is [" << d_vec.numRows() << " x " << d_vec.numCols() << " x " << d_vec.numMats()
                      << "]"
                      << " but risk has dimensions [" << m_dimension << " x " << 1 << " x " << 1 << "]\n";
            throw std::invalid_argument("DTensor and risk dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~CoherentRisk() {}

    virtual void preE(DTensor<real_t> &d_vec) = 0;  ///< Pre multiply given device vector with E matrix
    virtual void preF(DTensor<real_t> &d_vec) = 0;  ///< Pre multiply given device vector with F matrix
    virtual ConvexCone &cone() = 0;

    virtual void vecAddB(DTensor<real_t> &d_vec) = 0;

};


/**
 * Average Value at Risk (AVaR)
 * - parameters: alpha
 * - matrix F is not needed
*/
class AVaR : public CoherentRisk {

protected:
    real_t m_alpha = 0;
    DTensor<size_t> &m_d_chFrom;
    DTensor<real_t> &m_d_condProbs;
    NonnegativeOrthantCone m_nnoc;
    ZeroCone m_zero;
    Cartesian m_riskConeK;

public:
    explicit AVaR(size_t node,
                  real_t alpha,
                  DTensor<size_t> &d_numChildren,
                  DTensor<size_t> &d_childFrom,
                  DTensor<real_t> &d_condProbs) :
        CoherentRisk(node, d_numChildren),
        m_alpha(alpha),
        m_d_chFrom(d_childFrom),
        m_d_condProbs(d_condProbs),
        m_nnoc(m_numCh * 2),
        m_zero(1),
        m_riskConeK() {
        m_riskConeK.addCone(m_nnoc);
        m_riskConeK.addCone(m_zero);
    }

    void preE(DTensor<real_t> &d_vec) { return; }

    void preF(DTensor<real_t> &d_vec) { return; }

    ConvexCone &cone() { return m_riskConeK; }

    void vecAddB(DTensor<real_t> &d_vec) {
        dimension_check(d_vec);
        d_avarVecAddB<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(
            d_vec.raw(), m_node, m_d_numCh.raw(), m_d_chFrom.raw(), m_d_condProbs.raw());
    }

};


#endif
