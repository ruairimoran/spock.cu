#ifndef __CUBLAS_CONTEXT__
#define __CUBLAS_CONTEXT__


class Context {

    private:
        cublasHandle_t cublasHandle;
        
    public:
        explicit Context() noexcept { cublasCreate(&cublasHandle); }

        virtual ~Context() noexcept { cublasDestroy(cublasHandle); }

        cublasHandle_t &handle() { return cublasHandle; }
        
};

#endif
