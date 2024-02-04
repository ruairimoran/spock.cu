#include "../include/stdgpu.h"
#include <iostream>


__global__ void maxWithZero(real_t* vec, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) vec[i] = max(0., vec[i]);
}


__global__ void setToZero(real_t* vec, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) vec[i] = 0.;
}


static void projectOnZero(real_t* vec, size_t size) {
    setToZero<<<1, size>>>(vec, size);
}


__global__ void projectOnSoc(real_t* vec, size_t size, real_t nrm, real_t last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size - 1) vec[i] = last * (vec[i] / nrm);
    if (i == size - 1) vec[i] = last;
}


class ConvexCone {

    protected:
        Context& m_context;
        explicit ConvexCone(Context& context) : m_context(context) {};

    public:
        virtual void projectOnCone(real_t* vec, size_t size) = 0;
        virtual void projectOnDual(real_t* vec, size_t size) = 0;
};


/**
 * The Real cone
 * - the set is R^n
 * - the dual is the Zero cone
*/
class Real : public ConvexCone{
    
    public:
        Real(Context& context): ConvexCone(context) {}

        void projectOnCone(real_t* vec, size_t size) {
            // Do nothing!
        }
        void projectOnDual(real_t* vec, size_t size) {
            projectOnZero(vec, size);
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

        void projectOnCone(real_t* vec, size_t size) {
            projectOnZero(vec, size);
        }
        void projectOnDual(real_t* vec, size_t size) {
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

        void projectOnCone(real_t* vec, size_t size) {
            maxWithZero<<<1, size>>>(vec, size);
        }
        void projectOnDual(real_t* vec, size_t size) {
            projectOnCone(vec, size);
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

        void projectOnCone(real_t* vec, size_t size) {
            /** Sanity check */
            if (size < 2) {
                std::invalid_argument("Attempt to project onto a second order cone with a number.");
            }
            /** Determine the 2-norm of the first (size - 1) elements of vec */
            real_t nrm;
            cublasDnrm2(m_context.handle(), size - 1, vec, 1, &nrm);
            real_t lastElement = vec[size - 1];
            if (nrm <= lastElement) {
                // Do nothing!
            } else if (nrm <= -lastElement) {
                setToZero(vec, size);
            } else {
                real_t projectLastElement = (nrm + lastElement) / 2;
                projectOnSoc<<<1, size>>>(vec, size, nrm, projectLastElement);
            }
        }
        void projectOnDual(real_t* vec, size_t size) {
            projectOnCone(vec, size);
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

        void projectOnCone(real_t* vec, size_t size) {
            // complete!
        }
        void projectOnDual(real_t* vec, size_t size) {
            // complete!
        }
};
