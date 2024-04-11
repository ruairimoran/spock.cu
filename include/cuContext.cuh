#ifndef __CUBLAS_CONTEXT__
#define __CUBLAS_CONTEXT__


class Context {

    private:
        cublasHandle_t m_cublasHandle;
        cusolverDnHandle_t m_cusolverHandle;
        
    public:
        explicit Context() noexcept {
            cublasCreate(&m_cublasHandle);
            cusolverDnCreate(&m_cusolverHandle);
        }

        virtual ~Context() noexcept {
            cublasDestroy(m_cublasHandle);
            cusolverDnDestroy(m_cusolverHandle);
        }

        cublasHandle_t& blas() { return m_cublasHandle; }
        cusolverDnHandle_t& solver() { return m_cusolverHandle; }
};


#endif
