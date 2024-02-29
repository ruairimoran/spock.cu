#ifndef __CUBLAS_CONTEXT__
#define __CUBLAS_CONTEXT__


class Context {

    private:
        cublasHandle_t m_cublasHandle;
        
    public:
        explicit Context() noexcept { cublasCreate(&m_cublasHandle); }

        virtual ~Context() noexcept { cublasDestroy(m_cublasHandle); }

        cublasHandle_t& handle() { return m_cublasHandle; }
        
};

#endif
