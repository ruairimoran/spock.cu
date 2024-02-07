#include "../include/stdgpu.h"
#include <iostream>


static __global__ void maxWithZero(real_t* vec, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}


static __global__ void setToZero(real_t* vec, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}


static __global__ void projectOnSocElse(real_t* vec, size_t n, real_t nrm, real_t last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] = last * (vec[i] / nrm);
    if (i == n - 1) vec[i] = last;
}


static __global__ void projectOnSoc(real_t* vec, size_t n, real_t nrm) {
    if (nrm <= vec[n-1]) {
        // Do nothing!
    } else if (nrm <= -vec[n-1]) {
        setToZero<<<1, n>>>(vec, n);
    } else {
        real_t avg = (nrm + vec[n-1]) / 2.;
        projectOnSocElse<<<1, n>>>(vec, n, nrm, avg);
    }
}


class ConvexCone {

    protected:
        Context& m_context;
        explicit ConvexCone(Context& context) : m_context(context) {}

    public:
        virtual ~ConvexCone() {}

        virtual void projectOnCone(real_t* d_vec, size_t n) = 0;
        virtual void projectOnDual(real_t* d_vec, size_t n) = 0;
};


/**
 * The Universe cone (Univ)
 * - the set is R^n
 * - the dual is the Zero cone
*/
class UniverseCone : public ConvexCone {
    
    public:
        UniverseCone(Context& context) : ConvexCone(context) {}

        void projectOnCone(real_t* d_vec, size_t n) {
            // Do nothing!
        }
        void projectOnDual(real_t* d_vec, size_t n) {
            setToZero<<<1, n>>>(d_vec, n);
        }
};


/**
 * The Zero cone (Zero)
 * - the set is {0}
 * - the dual is the Universe cone
*/
class ZeroCone : public ConvexCone {
    
    public:
        ZeroCone(Context& context) : ConvexCone(context) {}

        void projectOnCone(real_t* d_vec, size_t n) {
            setToZero<<<1, n>>>(d_vec, n);
        }
        void projectOnDual(real_t* d_vec, size_t n) {
            // Do nothing!
        }
};


/**
 * The Nonnegative Orthant cone (NnOC)
 * - the set is R^n_+
 * - the cone is self dual
*/
class NonnegativeOrthantCone : public ConvexCone {
    
    public:
        NonnegativeOrthantCone(Context& context) : ConvexCone(context) {}

        void projectOnCone(real_t* d_vec, size_t n) {
            maxWithZero<<<1, n>>>(d_vec, n);
        }
        void projectOnDual(real_t* d_vec, size_t n) {
            projectOnCone(d_vec, n);
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
        SecondOrderCone(Context& context) : ConvexCone(context) {}

        void projectOnCone(real_t* d_vec, size_t n) {
            /** Sanity check */
            if (n < 2) {
                std::invalid_argument("Attempt to project onto a second order cone with a number.");
            }
            /** Determine the 2-norm of the first (n - 1) elements of vec */
            real_t nrm;
            cublasDnrm2(m_context.handle(), n-1, d_vec, 1, &nrm);
            projectOnSoc<<<1, 1>>>(d_vec, n, nrm);
        }
        void projectOnDual(real_t* d_vec, size_t n) {
            projectOnCone(d_vec, n);
        }
};


/**
 * A Cartesian cone (Cart)
 * - the set is a Cartesian product of cones (cone x cone x ...)
 * - the dual is the concatenation of the dual of each constituent cone 
*/
class Cartesian : public ConvexCone {

    private:
        std::vector<ConvexCone*>& m_cones;
        std::vector<size_t> m_sizes;
        size_t m_index;
    
    public:
        Cartesian(Context& context, std::vector<ConvexCone*>& cones, std::vector<size_t> sizes) : 
            ConvexCone(context), m_cones(cones), m_sizes(sizes) {}

        void projectOnCone(real_t* d_vec, size_t n=0) {
            m_index = 0;
            for (size_t i=0; i<m_cones.size(); i++) {
                m_cones[i]->projectOnCone(&d_vec[m_index], m_sizes[i]);
                m_index += m_sizes[i];
            }
        }
        void projectOnDual(real_t* d_vec, size_t n=0) {
            m_index = 0;
            for (size_t i=0; i<m_cones.size(); i++) {
                m_cones[i]->projectOnDual(&d_vec[m_index], m_sizes[i]);
                m_index += m_sizes[i];
            }
        }
};
