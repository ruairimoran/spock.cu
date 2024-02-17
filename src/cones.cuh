#ifndef __CONES__
#define __CONES__
#include "../include/stdgpu.h"


__global__ void d_maxWithZero(real_t* vec, size_t n);
__global__ void d_setToZero(real_t* vec, size_t n);
__global__ void d_projectOnSoc(real_t* vec, size_t n, real_t nrm, real_t scaling);


class ConvexCone {

    protected:
        Context& m_context;
        size_t m_dimension = 0;

        explicit ConvexCone(Context& context, size_t dim) : m_context(context), m_dimension(dim) {}

        bool dimension_check(DeviceVector<real_t>& d_vec) {
            if (d_vec.capacity() != m_dimension) {
                std::cerr << "DeviceVector has capacity " << d_vec.capacity()
                          << " but cone has dimension " << m_dimension << std::endl;
                throw std::invalid_argument("DeviceVector and cone dimension mismatch");
            }
            return true;
        }

    public:
        virtual ~ConvexCone() {}
        virtual void projectOnCone(DeviceVector<real_t>& d_vec) = 0;
        virtual void projectOnDual(DeviceVector<real_t>& d_vec) = 0;
        size_t dimension() { return m_dimension; }

};


/**
 * A null cone (Null)
 * - used as placeholder
*/
class NullCone : public ConvexCone {
    
    public:
        explicit NullCone(Context& context, size_t dim) : ConvexCone(context, dim) {}

        void projectOnCone(DeviceVector<real_t>& d_vec) {
            throw std::invalid_argument("Cannot project on null cone");
        }
        void projectOnDual(DeviceVector<real_t>& d_vec) {
            projectOnCone(d_vec);
        }

};


/**
 * The Universe cone (Univ)
 * - the set is R^n
 * - the dual is the Zero cone
*/
class UniverseCone : public ConvexCone {
    
    public:
        explicit UniverseCone(Context& context, size_t dim) : ConvexCone(context, dim) {}

        void projectOnCone(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            return;  // Do nothing!
        }
        void projectOnDual(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            d_setToZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.get(), m_dimension);
        }

};


/**
 * The Zero cone (Zero)
 * - the set is {0}
 * - the dual is the Universe cone
*/
class ZeroCone : public ConvexCone {
    
    public:
        explicit ZeroCone(Context& context, size_t dim) : ConvexCone(context, dim) {}

        void projectOnCone(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            d_setToZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.get(), m_dimension);
        }
        void projectOnDual(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            return;  // Do nothing!
        }

};


/**
 * The Nonnegative Orthant cone (NnOC)
 * - the set is R^n_+
 * - the cone is self dual
*/
class NonnegativeOrthantCone : public ConvexCone {
    
    public:
        explicit NonnegativeOrthantCone(Context& context, size_t dim) : ConvexCone(context, dim) {}

        void projectOnCone(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            d_maxWithZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.get(), m_dimension);
        }
        void projectOnDual(DeviceVector<real_t>& d_vec) {
            projectOnCone(d_vec);
        }

};


/**
 * The Second Order cone (SOC)
 * - the set is R^n_2
 * - the cone is self dual
 * - this projection follows [page 184, Section 6.3.2] of
 * > Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and trends® in Optimization, 1(3), 127-239.
*/
class SecondOrderCone : public ConvexCone {

    public:
        explicit SecondOrderCone(Context& context, size_t dim) : ConvexCone(context, dim) {}

        void projectOnCone(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            /** Determine the 2-norm of the first (n - 1) elements of d_vec */
            real_t nrm;
            cublasDnrm2(m_context.handle(), m_dimension-1, d_vec.get(), 1, &nrm);
            float vecLastElement = d_vec.fetchElementFromDevice(m_dimension - 1);
            if (nrm <= vecLastElement) {
                return;  // Do nothing!
            } else if (nrm <= -vecLastElement) {
                d_setToZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.get(), m_dimension);
            } else {
                real_t scaling = (nrm + vecLastElement) / (2. * nrm);
                d_projectOnSoc<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.get(), m_dimension, nrm, scaling);
            }
        }
        void projectOnDual(DeviceVector<real_t>& d_vec) {
            projectOnCone(d_vec);
        }

};


/**
 * A Cartesian cone (Cart)
 * - the set is a Cartesian product of cones (cone x cone x ...)
 * - the dual is the concatenation of the dual of each constituent cone 
*/
class Cartesian : public ConvexCone {

    private:
        std::vector<ConvexCone*> m_cones;
    
    public:
        explicit Cartesian(Context& context) : ConvexCone(context, 0) {}

        void addCone(ConvexCone& cone){
            m_cones.push_back(&cone);
            m_dimension += cone.dimension();
        }

        void projectOnCone(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            size_t start = 0;
            for (ConvexCone* set: m_cones) {
                size_t coneDim = set->dimension();
                size_t end = start + coneDim - 1;
                DeviceVector<real_t> vecSlice(d_vec, start, end);
                set->projectOnCone(vecSlice);
                start += coneDim;
            }
        }
        void projectOnDual(DeviceVector<real_t>& d_vec) {
            dimension_check(d_vec);
            size_t start = 0;
            for (ConvexCone* set: m_cones) {
                size_t coneDim = set->dimension();
                size_t end = start + coneDim - 1;
                DeviceVector<real_t> vecSlice(d_vec, start, end);
                set->projectOnDual(vecSlice);
                start += coneDim;
            }
        }

};

#endif
