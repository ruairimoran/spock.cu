#include "../include/stdgpu.h"
#include <iostream>


__global__ void maxWithZero(real_t* vec, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) vec[i] = max(0., vec[i]);
}


class ConvexCone {

    protected:
        Context& m_context;
        explicit ConvexCone(Context& context) : m_context(context) {};

    public:
        virtual void project_on_cone(real_t* vec, size_t size) = 0;
        virtual void project_on_dual(real_t* vec, size_t size) = 0;
};


class Real : public ConvexCone{
    
    public:
        Real(Context& context): ConvexCone(context) {}

        void project_on_cone(real_t* vec, size_t size) {
            // complete!
        }
        void project_on_dual(real_t* vec, size_t size) {
            // complete!
        }
};


class Zero : public ConvexCone{
    
    public:
        Zero(Context& context): ConvexCone(context) {}

        void project_on_cone(real_t* vec, size_t size) {
            // complete!
        }
        void project_on_dual(real_t* vec, size_t size) {
            // complete!
        }
};


class NonnegativeOrthant : public ConvexCone{
    
    public:
        NonnegativeOrthant(Context& context): ConvexCone(context) {}

        void project_on_cone(real_t* vec, size_t size) {
            maxWithZero<<<1, size>>>(vec, size);
        }
        void project_on_dual(real_t* vec, size_t size) {
            // complete!
        }
};


class SOC : public ConvexCone{

    public:
        explicit SOC(Context& context) : ConvexCone(context) {}

        void project_on_cone(real_t* vec, size_t size) {
            /* Determine the norm of the first n-1 elements of x */
            real_t nrm;
            cublasSnrm2(m_context.handle(), size - 1, vec, 1, &nrm);
            std::cout << "||x|| = " << nrm << std::endl;
            // complete!
        }
        void project_on_dual(real_t* vec, size_t size) {
            // complete!
        }
};


class Cartesian : public ConvexCone{
    
    public:
        Cartesian(Context& context): ConvexCone(context) {}

        void project_on_cone(real_t* vec, size_t size) {
            // complete!
        }
        void project_on_dual(real_t* vec, size_t size) {
            // complete!
        }
};
