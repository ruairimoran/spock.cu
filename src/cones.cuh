#include "../include/stdgpu.h"
#include <iostream>


__global__ void maxWithZero(float* x, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = max(0., x[i]);
}


class ConvexCone {

    protected:
        Context& m_context;
        explicit ConvexCone(Context& context) : m_context(context) {};

    public:
        virtual void project_on_cone(float* x, size_t n) = 0;
        virtual void project_on_dual(float* x, size_t n) = 0;
};


class Real : public ConvexCone{
    
    public:
        Real(Context& context): ConvexCone(context) {}

        void project_on_cone(float* x, size_t n) {
            // complete!
        }
        void project_on_dual(float* x, size_t n) {
            // complete!
        }
};


class Zero : public ConvexCone{
    
    public:
        Zero(Context& context): ConvexCone(context) {}

        void project_on_cone(float* x, size_t n) {
            // complete!
        }
        void project_on_dual(float* x, size_t n) {
            // complete!
        }
};


class NonnegativeOrthant : public ConvexCone{
    
    public:
        NonnegativeOrthant(Context& context): ConvexCone(context) {}

        void project_on_cone(float* x, size_t n) {
            maxWithZero<<<1, n>>>(x, n);
        }
        void project_on_dual(float* x, size_t n) {
            // complete!
        }
};


class SOC : public ConvexCone{

    public:
        explicit SOC(Context& context) : ConvexCone(context) {}

        void project_on_cone(float* x, size_t n) {
            /* Determine the norm of the first n-1 elements of x */
            float nrm;
            cublasSnrm2(m_context.handle(), n - 1, x, 1, &nrm);
            std::cout << "||x|| = " << nrm << std::endl;
            // complete!
        }
        void project_on_dual(float* x, size_t n) {
            // complete!
        }
};


class Cartesian : public ConvexCone{
    
    public:
        Cartesian(Context& context): ConvexCone(context) {}

        void project_on_cone(float* x, size_t n) {
            // complete!
        }
        void project_on_dual(float* x, size_t n) {
            // complete!
        }
};
