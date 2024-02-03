class Context {
private:
    cublasHandle_t cublasHandle;
public:
    explicit Context() noexcept {
        cublasCreate(&cublasHandle);
        std::cout << "CREATED CONTEXT!" << std::endl;
    }

    virtual ~Context() noexcept {
        cublasDestroy(cublasHandle);
        std::cout << "DESTROYED CONTEXT!" << std::endl;
    }

    cublasHandle_t &handle() {
        return cublasHandle;
    }

};
