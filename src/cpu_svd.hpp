#ifndef SPOCK_CPU_SVD_HPP
#define SPOCK_CPU_SVD_HPP

#include <memory>

// CPU-only implementation is compiled separately so CUDA translation units do
// not parse Eigen. Inputs and outputs use column-major storage; right() is V.
template<typename T>
class CpuSvd {
    struct Impl;
    std::unique_ptr<Impl> m_impl;
public:
    explicit CpuSvd(int size);
    ~CpuSvd();
    CpuSvd(const CpuSvd &) = delete;
    CpuSvd &operator=(const CpuSvd &) = delete;
    bool factorise(const T *matrix);
    const T *left() const;
    const T *right() const;
    const T *singularValues() const;
};

extern template class CpuSvd<float>;
extern template class CpuSvd<double>;

namespace spock::detail {
// Eigen's allocation guard is used only by serial unit tests, never changed by
// production factorise(). Return the old value so tests can restore it safely.
bool setCpuSvdAllocationAllowedForTesting(bool allowed);
}

#endif
