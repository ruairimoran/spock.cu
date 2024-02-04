#include "../include/stdgpu.h"
#include <iostream>


__global__ void maxWithZero(real_t* vec, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = max(0., vec[i]);
}


__global__ void setToZero(real_t* vec, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] = 0.;
}


__global__ void projectOnSoc(real_t* vec, size_t n, real_t nrm, real_t last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) vec[i] = last * (vec[i] / nrm);
    if (i == n - 1) vec[i] = last;
}


class ConvexCone {

    protected:
        Context& m_context;
        explicit ConvexCone(Context& context) : m_context(context) {};

    public:
        virtual void projectOnCone(DeviceVector<real_t>& vec) = 0;
        virtual void projectOnDual(DeviceVector<real_t>& vec) = 0;
};


/**
 * The Real cone
 * - the set is R^n
 * - the dual is the Zero cone
*/
class Real : public ConvexCone{
    
    public:
        Real(Context& context): ConvexCone(context) {}

        void projectOnCone(DeviceVector<real_t>& vec) {
            // Do nothing!
        }
        void projectOnDual(DeviceVector<real_t>& vec) {
            size_t n = vec.capacity();
            setToZero<<<1, n>>>(vec.get(), n);
        }
};


/**
 * The Zero cone
 * - the set is {0}
 * - the dual is the Real cone
*/
class Zero : public ConvexCone{
    
    public:
        Zero(Context& context): ConvexCone(context) {}

        void projectOnCone(DeviceVector<real_t>& vec) {
            size_t n = vec.capacity();
            setToZero<<<1, n>>>(vec.get(), n);
        }
        void projectOnDual(DeviceVector<real_t>& vec) {
            // Do nothing!
        }
};


/**
 * The Nonnegative Orthant cone
 * - the set is R^n_+
 * - the cone is self dual
*/
class NonnegativeOrthant : public ConvexCone{
    
    public:
        NonnegativeOrthant(Context& context): ConvexCone(context) {}

        void projectOnCone(DeviceVector<real_t>& vec) {
            size_t n = vec.capacity();
            maxWithZero<<<1, n>>>(vec.get(), n);
        }
        void projectOnDual(DeviceVector<real_t>& vec) {
            projectOnCone(vec);
        }
};


/**
 * The Second Order cone (SOC)
 * - the set is R^n_2
 * - the cone is self dual
*/
class SOC : public ConvexCone{

    public:
        explicit SOC(Context& context) : ConvexCone(context) {}

        void projectOnCone(DeviceVector<real_t>& vec) {
            size_t n = vec.capacity();
            /** Sanity check */
            if (n < 2) {
                std::invalid_argument("Attempt to project onto a second order cone with a number.");
            }
            /** Get */
            real_t* vecPtr = vec.get();
            /** Determine the 2-norm of the first (n - 1) elements of vec */
            real_t nrm;
            cublasDnrm2(m_context.handle(), n-1, vecPtr, 1, &nrm);
            real_t last;
            vec.download(&last, n-1);
            if (nrm <= last) {
                // Do nothing!
            } else if (nrm <= -last) {
                setToZero<<<1, n>>>(vecPtr, n);
            } else {
                real_t avg = (nrm + last) / 2.;
                projectOnSoc<<<1, n>>>(vecPtr, n, nrm, avg);
            }
        }
        void projectOnDual(DeviceVector<real_t>& vec) {
            projectOnCone(vec);
        }
};


/**
 * A Cartesian cone
 * - the set is a Cartesian product of cones (cone x cone x ...)
 * - the dual is the concatenation of the dual of each constituent cone 
*/
class Cartesian : public ConvexCone{
    
    public:
        Cartesian(Context& context): ConvexCone(context) {}

        void projectOnCone(DeviceVector<real_t>& vec) {
            // todo!
        }
        void projectOnDual(DeviceVector<real_t>& vec) {
            // todo!
        }
};
