#ifndef __RISKS__
#define __RISKS__
#include "../include/stdgpu.h"
#include "cones.cuh"


__global__ void vecAddB(real_t* vec, size_t ch, size_t* probs);


/**
 * Base class for a coherent risk
 * that can be described by the tuple
 * (E, F, K, b)
 */
class CoherentRisk {

    protected:
        Context& m_context;
        size_t m_numCh = 0;
        size_t m_dimension = 0;

        explicit CoherentRisk(Context& context, size_t numChildren, size_t rows) :
            m_context(context), m_numCh(numChildren), m_dimension(rows) {}

        bool dimension_check(DeviceVector<real_t>& d_vec) {
            if (d_vec.capacity() != m_dimension) {
                std::cerr << "DeviceVector has capacity " << d_vec.capacity()
                          << " but risk has dimension " << m_dimension << std::endl;
                throw std::invalid_argument("DeviceVector and risk size mismatch");
            }
            return true;
        }

    public:
        virtual ~CoherentRisk() {}
        virtual void preE(DeviceVector<real_t>& d_vec) = 0;
        virtual void preF(DeviceVector<real_t>& d_vec) = 0;
        virtual void vecAddB(DeviceVector<real_t>& d_vec) = 0;

};


/**
 * Average Value at Risk (AVaR)
 * - parameters: alpha
 * - matrix F is not needed
*/
class AVaR : public CoherentRisk {

    protected:
        real_t m_alpha = 0;
        DeviceVector<size_t>& m_condProbabilities;
        NonnegativeOrthantCone m_nnoc;
        ZeroCone m_zero;
        Cartesian m_riskConeK;

    public:
        explicit AVaR(Context& context, real_t a, size_t numCh, DeviceVector<size_t>& condProbs) :
            CoherentRisk(context, numCh, numCh*2+1),
            m_alpha(a), m_condProbabilities(condProbs),
            m_nnoc(context, numCh*2), m_zero(context, 1), m_riskConeK(context) {
            m_riskConeK.addCone(m_nnoc);
            m_riskConeK.addCone(m_zero);
        }

        void vecAddB(DeviceVector<real_t>& d_vec) {
            ::vecAddB<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(
                    d_vec.get(), m_numCh, m_condProbabilities.get());
        }


//        eye = np.eye(self.__num_children);
//        riskMatEi = np.vstack((alpha*eye, -eye, np.ones((1, num_children))));
//        std::vector riskMatFi(2*chNum+1, 0.);

};


#endif
