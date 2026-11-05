#ifndef SPOCK_ANDERSON_QR_CUH
#define SPOCK_ANDERSON_QR_CUH

#include "../include/gpu.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

struct AndersonQrOptions {
    bool incremental = true;  // Reuse the QR factors for regularised Anderson by default.
    size_t rebuildInterval = 64;
    size_t checkInterval = 16;
    double orthogonalityTolerance = 0.;  // Zero selects a precision-dependent default.

    void validate() const {
        if (!rebuildInterval || !checkInterval || !std::isfinite(orthogonalityTolerance) ||
            orthogonalityTolerance < 0 || orthogonalityTolerance >= 1)
            throw std::invalid_argument("Invalid Anderson QR options");
    }
};

struct AndersonQrStatistics {
    size_t rebuilds = 0;
    size_t updates = 0;
    size_t reuses = 0;
    size_t updateFallbacks = 0;
    size_t orthogonalityChecks = 0;
    double lastOrthogonalityError = 0.;
};

namespace anderson_qr_detail {

// Apply every deletion rotation in one pass over Q. Each thread owns a row,
// so columns can be overwritten in place without overlapping memory copies.
template<typename T>
__global__ void deleteOldestColumn(T *q, int rows, int memory, const T *rotations) {
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= size_t(rows)) return;
    T left = q[row];
    for (int j = 0; j < memory - 1; ++j) {
        const T right = q[(j + 1) * rows + row];
        const T cosine = rotations[2 * j], sine = rotations[2 * j + 1];
        q[j * rows + row] = cosine * left + sine * right;
        left = -sine * left + cosine * right;
    }
}

template<typename T>
__global__ void normaliseColumn(const T *column, T *q, int rows, T inverseNorm) {
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < size_t(rows)) q[row] = column[row] * inverseNorm;
}

}  // namespace anderson_qr_detail

// Maintains Y_oldest_first = Q R while the caller stores Y newest first.
// Only the small R columns are reversed on output; physical history and the
// Anderson coefficients retain their existing newest-first convention.
template<typename T>
class AndersonRollingQr {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    static int validateWorkspaces(const DTensor<T> &y, const DTensor<T> &q, const DTensor<T> &work) {
        const size_t rows = y.numRows(), memory = y.numCols();
        if (!memory || rows < memory || rows > size_t(std::numeric_limits<int>::max()) / memory ||
            y.numMats() != 1 || q.numRows() != rows || q.numCols() != memory || q.numMats() != 1 ||
            work.numRows() != rows || work.numCols() != 1 || work.numMats() != 1)
            throw std::invalid_argument("Invalid rolling QR workspaces");
        return int(rows);
    }

    DTensor<T> &m_y, &m_q, &m_work;
    int m_rows, m_memory, m_workspaceSize = 0;
    double m_rankTolerance;
    AndersonQrOptions m_options;
    AndersonQrStatistics m_statistics;
    DTensor<T> m_tau, m_projection, m_rotationsDevice, m_gram;
    DTensor<int> m_info;
    std::unique_ptr<DTensor<T>> m_workspace;
    std::vector<T> m_r, m_shiftedR, m_rotations, m_projectionHost, m_gramHost;
    cudaStream_t m_stream{};
    size_t m_generation = 0, m_factorGeneration = 0, m_updatesSinceRebuild = 0;
    bool m_valid = false, m_updatable = false;

    cublasHandle_t blas() const { return Session::getInstance().cuBlasHandle(); }
    cusolverDnHandle_t solver() const { return Session::getInstance().cuSolverHandle(); }

    bool solverSucceeded() {
        int info;
        gpuErrChk(cudaMemcpy(&info, m_info.raw(), sizeof(info), cudaMemcpyDeviceToHost));
        return info == 0;
    }

    void project(const T *vector, int columns, T *result) {
        const T one = 1, zero = 0;
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cublasDgemv(blas(), CUBLAS_OP_T, m_rows, columns, &one,
                m_q.raw(), m_rows, vector, 1, &zero, result, 1));
        } else {
            gpuErrChk(cublasSgemv(blas(), CUBLAS_OP_T, m_rows, columns, &one,
                m_q.raw(), m_rows, vector, 1, &zero, result, 1));
        }
    }

    double columnNorm(int column) const {
        double norm = 0.;
        for (int row = 0; row < m_memory; ++row) norm = std::hypot(norm, double(m_r[column * m_memory + row]));
        return norm;
    }

    bool wellConditionedForUpdate() const {
        double scale = 0.;
        for (int col = 0; col < m_memory; ++col) scale = std::max(scale, columnNorm(col));
        if (!std::isfinite(scale) || scale == 0) return false;
        for (int col = 0; col < m_memory; ++col)
            if (!std::isfinite(double(m_r[col * m_memory + col])) ||
                std::abs(double(m_r[col * m_memory + col])) <= m_rankTolerance * scale) return false;
        return true;
    }

    bool rebuild() {
        ++m_statistics.rebuilds;
        m_valid = false;
        for (int col = 0; col < m_memory; ++col)
            gpuErrChk(cudaMemcpy(m_q.raw() + col * m_rows,
                m_y.raw() + (m_memory - 1 - col) * m_rows, m_rows * sizeof(T), cudaMemcpyDeviceToDevice));
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDgeqrf(solver(), m_rows, m_memory, m_q.raw(), m_rows,
                m_tau.raw(), m_workspace->raw(), m_workspaceSize, m_info.raw()));
        } else {
            gpuErrChk(cusolverDnSgeqrf(solver(), m_rows, m_memory, m_q.raw(), m_rows,
                m_tau.raw(), m_workspace->raw(), m_workspaceSize, m_info.raw()));
        }
        if (!solverSucceeded()) return false;
        gpuErrChk(cudaMemcpy2D(m_r.data(), m_memory * sizeof(T), m_q.raw(), m_rows * sizeof(T),
            m_memory * sizeof(T), m_memory, cudaMemcpyDeviceToHost));
        for (int col = 0; col < m_memory; ++col)
            for (int row = col + 1; row < m_memory; ++row) m_r[col * m_memory + row] = 0;
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDorgqr(solver(), m_rows, m_memory, m_memory, m_q.raw(), m_rows,
                m_tau.raw(), m_workspace->raw(), m_workspaceSize, m_info.raw()));
        } else {
            gpuErrChk(cusolverDnSorgqr(solver(), m_rows, m_memory, m_memory, m_q.raw(), m_rows,
                m_tau.raw(), m_workspace->raw(), m_workspaceSize, m_info.raw()));
        }
        if (!solverSucceeded()) return false;
        m_valid = true;
        m_updatable = wellConditionedForUpdate();
        m_updatesSinceRebuild = 0;
        return true;
    }

    bool appendNewestColumn() {
        // Two-pass classical Gram--Schmidt uses GEMV for the whole thin basis
        // rather than launching a dot and an AXPY separately for each column.
        gpuErrChk(cudaMemcpy(m_work.raw(), m_y.raw(), m_rows * sizeof(T), cudaMemcpyDeviceToDevice));
        const int retained = m_memory - 1;
        const T minusOne = -1, one = 1;
        for (int pass = 0; pass < 2 && retained; ++pass) {
            project(m_work.raw(), retained, m_projection.raw());
            gpuErrChk(cudaMemcpy(m_projectionHost.data(), m_projection.raw(), retained * sizeof(T), cudaMemcpyDeviceToHost));
            for (int col = 0; col < retained; ++col)
                m_r[retained * m_memory + col] += m_projectionHost[col];
            if constexpr (std::is_same_v<T, double>) {
                gpuErrChk(cublasDgemv(blas(), CUBLAS_OP_N, m_rows, retained, &minusOne,
                    m_q.raw(), m_rows, m_projection.raw(), 1, &one, m_work.raw(), 1));
            } else {
                gpuErrChk(cublasSgemv(blas(), CUBLAS_OP_N, m_rows, retained, &minusOne,
                    m_q.raw(), m_rows, m_projection.raw(), 1, &one, m_work.raw(), 1));
            }
        }
        const T norm = m_work.normF();
        m_r[retained * m_memory + retained] = norm;
        // A nearly dependent insertion is handled by a fresh Householder QR,
        // which also supplies an orthonormal completion for deficient history.
        if (!wellConditionedForUpdate()) return false;
        const T inverse = T(1) / norm;
        if (!std::isfinite(double(inverse))) return false;
        anderson_qr_detail::normaliseColumn<<<1 + (m_rows - 1) / 256, 256, 0, m_stream>>>(
            m_work.raw(), m_q.raw() + retained * m_rows, m_rows, inverse);
        gpuErrChk(cudaGetLastError());
        return true;
    }

    bool update() {
        std::fill(m_shiftedR.begin(), m_shiftedR.end(), T(0));
        for (int col = 0; col < m_memory - 1; ++col)
            for (int row = 0; row < m_memory; ++row)
                m_shiftedR[col * m_memory + row] = m_r[(col + 1) * m_memory + row];
        for (int i = 0; i < m_memory - 1; ++i) {
            const double a = m_shiftedR[i * m_memory + i], b = m_shiftedR[i * m_memory + i + 1];
            const double norm = std::hypot(a, b);
            if (!std::isfinite(norm) || norm == 0) return false;
            const T cosine = T(a / norm), sine = T(b / norm);
            m_rotations[2 * i] = cosine; m_rotations[2 * i + 1] = sine;
            for (int col = i; col < m_memory - 1; ++col) {
                const T upper = m_shiftedR[col * m_memory + i], lower = m_shiftedR[col * m_memory + i + 1];
                m_shiftedR[col * m_memory + i] = cosine * upper + sine * lower;
                m_shiftedR[col * m_memory + i + 1] = -sine * upper + cosine * lower;
            }
            m_shiftedR[i * m_memory + i + 1] = 0;
        }
        m_r.swap(m_shiftedR);
        if (m_memory > 1) {
            gpuErrChk(cudaMemcpy(m_rotationsDevice.raw(), m_rotations.data(), m_rotations.size() * sizeof(T), cudaMemcpyHostToDevice));
            anderson_qr_detail::deleteOldestColumn<<<1 + (m_rows - 1) / 256, 256, 0, m_stream>>>(
                m_q.raw(), m_rows, m_memory, m_rotationsDevice.raw());
            gpuErrChk(cudaGetLastError());
        }
        if (!appendNewestColumn()) return false;
        ++m_updatesSinceRebuild;
        if (m_updatesSinceRebuild % m_options.checkInterval == 0 && !checkOrthogonality()) return false;
        return true;
    }

public:
    AndersonRollingQr(DTensor<T> &y, DTensor<T> &q, DTensor<T> &work,
                      AndersonQrOptions options, double rankTolerance)
        : m_y(y), m_q(q), m_work(work), m_rows(validateWorkspaces(y, q, work)), m_memory(int(y.numCols())),
          m_rankTolerance(rankTolerance), m_options(options), m_tau(m_memory), m_projection(m_memory),
          m_rotationsDevice(std::max(1, 2 * (m_memory - 1))), m_gram(m_memory, m_memory), m_info(1),
          m_r(m_memory * m_memory), m_shiftedR(m_memory * m_memory),
          m_rotations(2 * (m_memory - 1)), m_projectionHost(m_memory), m_gramHost(m_memory * m_memory) {
        m_options.validate();
        if (m_options.orthogonalityTolerance == 0)
            m_options.orthogonalityTolerance = std::is_same_v<T, double> ? 1e-10 : 1e-4;
        gpuErrChk(cublasGetStream(blas(), &m_stream));
        int generateSize = 0;
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDgeqrf_bufferSize(solver(), m_rows, m_memory, m_q.raw(), m_rows, &m_workspaceSize));
            gpuErrChk(cusolverDnDorgqr_bufferSize(solver(), m_rows, m_memory, m_memory,
                m_q.raw(), m_rows, m_tau.raw(), &generateSize));
        } else {
            gpuErrChk(cusolverDnSgeqrf_bufferSize(solver(), m_rows, m_memory, m_q.raw(), m_rows, &m_workspaceSize));
            gpuErrChk(cusolverDnSorgqr_bufferSize(solver(), m_rows, m_memory, m_memory,
                m_q.raw(), m_rows, m_tau.raw(), &generateSize));
        }
        m_workspaceSize = std::max(m_workspaceSize, generateSize);
        m_workspace = std::make_unique<DTensor<T>>(std::max(1, m_workspaceSize));
    }

    void historyChanged() { ++m_generation; }
    void invalidate() { m_valid = false; }
    void reset() {
        invalidate();
        m_generation = m_factorGeneration = m_updatesSinceRebuild = 0;
        m_statistics = {};
    }
    const AndersonQrStatistics &statistics() const { return m_statistics; }

    bool checkOrthogonality() {
        const T one = 1, zero = 0;
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cublasDgemm(blas(), CUBLAS_OP_T, CUBLAS_OP_N, m_memory, m_memory, m_rows,
                &one, m_q.raw(), m_rows, m_q.raw(), m_rows, &zero, m_gram.raw(), m_memory));
        } else {
            gpuErrChk(cublasSgemm(blas(), CUBLAS_OP_T, CUBLAS_OP_N, m_memory, m_memory, m_rows,
                &one, m_q.raw(), m_rows, m_q.raw(), m_rows, &zero, m_gram.raw(), m_memory));
        }
        gpuErrChk(cudaMemcpy(m_gramHost.data(), m_gram.raw(), m_gramHost.size() * sizeof(T), cudaMemcpyDeviceToHost));
        double error = 0.;
        for (int col = 0; col < m_memory; ++col)
            for (int row = 0; row < m_memory; ++row) {
                const double entry = double(m_gramHost[col * m_memory + row]) - (row == col ? 1. : 0.);
                if (!std::isfinite(entry)) error = INFINITY;
                else error = std::max(error, std::abs(entry));
            }
        ++m_statistics.orthogonalityChecks;
        m_statistics.lastOrthogonalityError = error;
        return error <= m_options.orthogonalityTolerance;
    }

    bool compress(DTensor<T> &residual, std::vector<T> &r, std::vector<T> &rhs) {
        bool success;
        if (m_valid && m_factorGeneration == m_generation) {
            ++m_statistics.reuses;
            success = true;
        } else if (m_valid && m_updatable && m_generation == m_factorGeneration + 1 &&
                   m_updatesSinceRebuild < m_options.rebuildInterval) {
            success = update();
            if (success) ++m_statistics.updates;
            else {
                ++m_statistics.updateFallbacks;
                success = rebuild();
            }
        } else success = rebuild();
        if (!success) { invalidate(); return false; }
        m_factorGeneration = m_generation;
        for (int col = 0; col < m_memory; ++col)
            for (int row = 0; row < m_memory; ++row)
                r[col * m_memory + row] = m_r[(m_memory - 1 - col) * m_memory + row];
        project(residual.raw(), m_memory, m_projection.raw());
        gpuErrChk(cudaMemcpy(rhs.data(), m_projection.raw(), m_memory * sizeof(T), cudaMemcpyDeviceToHost));
        return true;
    }
};

#endif
