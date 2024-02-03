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
    virtual void project(float* x, size_t n) = 0;
};

class NonnegativeOrthant : public ConvexCone{
private:
public:
    NonnegativeOrthant(Context& context): ConvexCone(context) {}

    void project(float* x, size_t n){
        maxWithZero<<<1, n>>>(x, n);
    }
};


class SOC : public ConvexCone{

public:
    explicit SOC(Context& context) : ConvexCone(context) {}

    void project(float* x, size_t n){
        /* Determine the norm of the first n-1 elements of x */
        float nrm;
        cublasSnrm2(m_context.handle(), n - 1, x, 1, &nrm);
        std::cout << "||x|| = " << nrm << std::endl;
    }
};
