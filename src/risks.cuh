#ifndef __RISKS__
#define __RISKS__

#include "../include/stdgpu.h"
#include "cones.cuh"


__global__ void d_avarVecAddB(real_t *vec, size_t node, size_t *numCh, size_t *chFrom, real_t *probs);


/**
 * Base class for a coherent risk
 * that can be described by the tuple
 * (E, F, K, b)
 */
class CoherentRisk {

protected:
    Context &m_context;
    size_t m_node = 0;
    DeviceVector<size_t> &m_d_numCh;
    size_t m_numCh = 0;
    size_t m_dimension = 0;

    explicit CoherentRisk(Context &context, size_t node, DeviceVector<size_t> &numCh) :
            m_context(context),
            m_node(node),
            m_d_numCh(numCh),
            m_numCh(m_d_numCh.fetchElementFromDevice(m_node)),
            m_dimension(m_numCh * 2 + 1) {}

    bool dimension_check(DeviceVector<real_t> &d_vec) {
        if (d_vec.capacity() != m_dimension) {
            std::cerr << "DeviceVector has capacity " << d_vec.capacity()
                      << " but risk has dimension " << m_dimension << std::endl;
            throw std::invalid_argument("DeviceVector and risk size mismatch");
        }
        return true;
    }

public:
    virtual ~CoherentRisk() {}

    virtual void preE(DeviceVector<real_t> &d_vec) = 0;  ///< Pre multiply given device vector with E matrix
    virtual void preF(DeviceVector<real_t> &d_vec) = 0;  ///< Pre multiply given device vector with F matrix
    virtual ConvexCone &cone() = 0;

    virtual void vecAddB(DeviceVector<real_t> &d_vec) = 0;

};


/**
 * Average Value at Risk (AVaR)
 * - parameters: alpha
 * - matrix F is not needed
*/
class AVaR : public CoherentRisk {

protected:
    real_t m_alpha = 0;
    DeviceVector<size_t> &m_d_chFrom;
    DeviceVector<real_t> &m_d_condProbs;
    NonnegativeOrthantCone m_nnoc;
    ZeroCone m_zero;
    Cartesian m_riskConeK;

public:
    explicit AVaR(Context &context,
                  size_t node,
                  real_t alpha,
                  DeviceVector<size_t> &d_numChildren,
                  DeviceVector<size_t> &d_childFrom,
                  DeviceVector<real_t> &d_condProbs) :
            CoherentRisk(context, node, d_numChildren),
            m_alpha(alpha),
            m_d_chFrom(d_childFrom),
            m_d_condProbs(d_condProbs),
            m_nnoc(context, m_numCh * 2),
            m_zero(context, 1),
            m_riskConeK(context) {
        m_riskConeK.addCone(m_nnoc);
        m_riskConeK.addCone(m_zero);
    }

    void preE(DeviceVector<real_t> &d_vec) { return; }

    void preF(DeviceVector<real_t> &d_vec) { return; }

    ConvexCone &cone() { return m_riskConeK; }

    void vecAddB(DeviceVector<real_t> &d_vec) {
        dimension_check(d_vec);
        d_avarVecAddB<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(
                d_vec.get(), m_node, m_d_numCh.get(), m_d_chFrom.get(), m_d_condProbs.get());
    }

};


#endif
