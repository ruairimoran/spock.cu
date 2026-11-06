#include "cpu_svd.hpp"
#include <stdexcept>

// Keep Eigen's runtime allocation check effective in Release builds as well.
#define eigen_assert(condition) do { if (!(condition)) throw std::logic_error("Eigen CPU SVD assertion failed"); } while (false)
#include <Eigen/SVD>

template<typename T>
struct CpuSvd<T>::Impl {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Matrix input;
    Eigen::JacobiSVD<Matrix, Eigen::NoQRPreconditioner> svd;
    explicit Impl(int size)
        : input(size,size), svd(size,size,Eigen::ComputeFullU | Eigen::ComputeFullV) {}
};

template<typename T>
CpuSvd<T>::CpuSvd(int size) {
    if (size < 1) throw std::invalid_argument("CPU SVD size must be positive");
    m_impl = std::make_unique<Impl>(size);
}

template<typename T>
CpuSvd<T>::~CpuSvd() = default;

template<typename T>
bool CpuSvd<T>::factorise(const T *matrix) {
    const auto size = m_impl->input.rows();
    // Copy into a preallocated matrix: passing Map directly would materialise
    // a temporary with this Eigen API. The factoriser retains all workspaces.
    m_impl->input = Eigen::Map<const typename Impl::Matrix>(matrix,size,size);
    m_impl->svd.compute(m_impl->input,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return m_impl->svd.info() == Eigen::Success;
}

template<typename T>
const T *CpuSvd<T>::left() const { return m_impl->svd.matrixU().data(); }
template<typename T>
const T *CpuSvd<T>::right() const { return m_impl->svd.matrixV().data(); }
template<typename T>
const T *CpuSvd<T>::singularValues() const { return m_impl->svd.singularValues().data(); }

template class CpuSvd<float>;
template class CpuSvd<double>;

bool spock::detail::setCpuSvdAllocationAllowedForTesting(bool allowed) {
    const bool previous = Eigen::internal::is_malloc_allowed();
    Eigen::internal::set_is_malloc_allowed(allowed);
    return previous;
}
