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


class ConvexCone {

    protected:
        Context& m_context;
        explicit ConvexCone(Context& context) : m_context(context) {};

    public:
        virtual void  projectOnCone(real_t* vec, size_t size) = 0;
        virtual void  projectOnDual(real_t* vec, size_t size) = 0;
};


/**
 * The Real cone
 * - the set is R^n
 * - the dual is the Zero cone
*/
class Real : public ConvexCone{
    
    public:
        Real(Context& context): ConvexCone(context) {}

        void  projectOnCone(real_t* vec, size_t size) {
            // Do nothing!
        }
        void  projectOnDual(real_t* vec, size_t size) {
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

        void  projectOnCone(real_t* vec, size_t size) {
            projectOnZero(vec, size);
        }
        void  projectOnDual(real_t* vec, size_t size) {
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

        void  projectOnCone(real_t* vec, size_t size) {
            maxWithZero<<<1, size>>>(vec, size);
        }
        void  projectOnDual(real_t* vec, size_t size) {
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

        void  projectOnCone(real_t* vec, size_t size) {
            /* Determine the norm of the first n-1 elements of x */
            real_t nrm;
            cublasSnrm2(m_context.handle(), size - 1, vec, 1, &nrm);
            std::cout << "||x|| = " << nrm << std::endl;
            // complete!
        }
        void  projectOnDual(real_t* vec, size_t size) {
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

        void  projectOnCone(real_t* vec, size_t size) {
            // complete!
        }
        void  projectOnDual(real_t* vec, size_t size) {
            // complete!
        }
};
