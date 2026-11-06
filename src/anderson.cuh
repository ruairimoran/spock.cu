#ifndef SPOCK_ANDERSON_CUH
#define SPOCK_ANDERSON_CUH

#include "../include/gpu.cuh"
#include "anderson_qr.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

struct AndersonOptions {
    bool enabled = true;  // Use the safeguarded, regularised fit by default.
    double regularization = 1e-8;
    double rankTolerance = 0.;  // Zero selects a precision-dependent default.
    double maxCoefficientNorm = 1e6;
    double maxDirectionRatio = 1e4;
    AndersonQrOptions qr;

    void validate() const {
        qr.validate();
        if (!std::isfinite(regularization) || regularization <= 0 ||
            !std::isfinite(rankTolerance) || rankTolerance < 0 || rankTolerance >= 1 ||
            !std::isfinite(maxCoefficientNorm) || maxCoefficientNorm <= 0 ||
            !std::isfinite(maxDirectionRatio) || maxDirectionRatio <= 0)
            throw std::invalid_argument("Invalid regularised Anderson options");
    }
};

enum class AndersonFitStatus {
    accepted, rankZero, nonfiniteHistory, solverFailure,
    nonfiniteSolution, coefficientLimit, directionLimit
};

inline const char *andersonStatusName(AndersonFitStatus status) {
    switch (status) {
        case AndersonFitStatus::accepted: return "accepted";
        case AndersonFitStatus::rankZero: return "rank_zero";
        case AndersonFitStatus::nonfiniteHistory: return "nonfinite_history";
        case AndersonFitStatus::solverFailure: return "solver_failure";
        case AndersonFitStatus::nonfiniteSolution: return "nonfinite_solution";
        case AndersonFitStatus::coefficientLimit: return "coefficient_limit";
        case AndersonFitStatus::directionLimit: return "direction_limit";
    }
    return "unknown";
}

struct AndersonFitInfo {
    AndersonFitStatus status = AndersonFitStatus::rankZero;
    std::vector<size_t> selectedColumns;
    std::vector<double> coefficients;
    double scale = 0.;
    double coefficientNorm = 0.;
    double directionRatio = 0.;

    bool accepted() const { return status == AndersonFitStatus::accepted; }
};

struct AndersonStatistics {
    size_t fits = 0;
    size_t accepted = 0;
    size_t filteredColumns = 0;
    size_t restarts = 0;
    size_t bootstrapSteps = 0;
    size_t coefficientRejections = 0;
    size_t directionRejections = 0;
    size_t numericalRejections = 0;
    size_t lastRank = 0;
    double lastScale = 0.;
    double lastCoefficientNorm = 0.;
    AndersonQrStatistics qr;
};

// A type-II Anderson fit with borrowed tall workspaces. All device storage is
// allocated at construction; compute() transfers only O(memory^2) fit data
// between host and device. C and Y must contain matching, newest-first columns,
// with C = delta(iterate) - delta(residual).
template<typename T>
class RegularisedAnderson {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    static int validateWorkspaces(const DTensor<T> &c, const DTensor<T> &y, const DTensor<T> &qr,
                                  const DTensor<T> &rhs, const DTensor<T> &gamma) {
        const size_t rows = y.numRows(), memory = y.numCols();
        if (memory == 0 || rows < memory || y.numMats() != 1 ||
            rows > size_t(std::numeric_limits<int>::max()) / memory ||
            c.numRows() != rows || c.numCols() != memory || c.numMats() != 1 ||
            qr.numRows() != rows || qr.numCols() != memory || qr.numMats() != 1 ||
            rhs.numRows() != rows || rhs.numCols() != 1 || rhs.numMats() != 1 ||
            gamma.numRows() != memory || gamma.numCols() != 1 || gamma.numMats() != 1)
            throw std::invalid_argument("Incompatible regularised Anderson workspaces");
        return int(rows);
    }

    static AndersonOptions validatedOptions(AndersonOptions options) {
        if (options.rankTolerance == 0) options.rankTolerance = std::is_same_v<T, float> ? 1e-4 : 1e-8;
        options.validate();
        return options;
    }

    struct SvdParameters {
        gesvdjInfo_t value{};
        SvdParameters() {
            gpuErrChk(cusolverDnCreateGesvdjInfo(&value));
            gpuErrChk(cusolverDnXgesvdjSetMaxSweeps(value, 100));
        }
        ~SvdParameters() { cusolverDnDestroyGesvdjInfo(value); }
    };

    DTensor<T> &m_c, &m_y, &m_qr, &m_rhs, &m_gamma;
    int m_rows, m_memory, m_qrWorkspaceSize = 0, m_svdWorkspaceSize = 0;
    AndersonOptions m_options;
    SvdParameters m_svdParameters;
    DTensor<T> m_tau, m_smallMatrix, m_u, m_v, m_singularValues;
    DTensor<int> m_info;
    std::unique_ptr<DTensor<T>> m_qrWorkspace, m_svdWorkspace;
    std::vector<T> m_rHost, m_rhsHost, m_uHost, m_vHost, m_sHost, m_packed, m_gammaHost;
    std::vector<double> m_basis, m_innovation, m_normalisedRhs, m_solution;
    AndersonFitInfo m_fit;
    std::unique_ptr<AndersonRollingQr<T>> m_rollingQr;
    AndersonQrStatistics m_fullQrStatistics;

    cusolverDnHandle_t handle() const { return Session::getInstance().cuSolverHandle(m_y.streamIdx()); }

    void allocateSolverWorkspaces() {
        int applySize = 0;
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDgeqrf_bufferSize(handle(), m_rows, m_memory, m_qr.raw(), m_rows, &m_qrWorkspaceSize));
            gpuErrChk(cusolverDnDormqr_bufferSize(handle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                m_rows, 1, m_memory, m_qr.raw(), m_rows, m_tau.raw(), m_rhs.raw(), m_rows, &applySize));
            gpuErrChk(cusolverDnDgesvdj_bufferSize(handle(), CUSOLVER_EIG_MODE_VECTOR, 0,
                m_memory, m_memory, m_smallMatrix.raw(), m_memory, m_singularValues.raw(),
                m_u.raw(), m_memory, m_v.raw(), m_memory, &m_svdWorkspaceSize, m_svdParameters.value));
        } else {
            gpuErrChk(cusolverDnSgeqrf_bufferSize(handle(), m_rows, m_memory, m_qr.raw(), m_rows, &m_qrWorkspaceSize));
            gpuErrChk(cusolverDnSormqr_bufferSize(handle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                m_rows, 1, m_memory, m_qr.raw(), m_rows, m_tau.raw(), m_rhs.raw(), m_rows, &applySize));
            gpuErrChk(cusolverDnSgesvdj_bufferSize(handle(), CUSOLVER_EIG_MODE_VECTOR, 0,
                m_memory, m_memory, m_smallMatrix.raw(), m_memory, m_singularValues.raw(),
                m_u.raw(), m_memory, m_v.raw(), m_memory, &m_svdWorkspaceSize, m_svdParameters.value));
        }
        // QR factorisation and application of Q' have different workspace needs.
        m_qrWorkspaceSize = std::max(m_qrWorkspaceSize, applySize);
        if (!m_options.qr.incremental)
            m_qrWorkspace = std::make_unique<DTensor<T>>(std::max(1, m_qrWorkspaceSize));
        m_svdWorkspace = std::make_unique<DTensor<T>>(std::max(1, m_svdWorkspaceSize));
        if (m_qrWorkspace) m_qrWorkspace->setStreamIdx(m_y.streamIdx());
        m_svdWorkspace->setStreamIdx(m_y.streamIdx());
    }

    bool solverSucceeded() {
        int info;
        spock::detail::copyToHost(&info, m_info.raw(), sizeof(info), spock::executionStream(m_y));
        return info == 0;
    }

    bool compressHistory(DTensor<T> &residual) {
        if (m_rollingQr) return m_rollingQr->compress(residual, m_rHost, m_rhsHost);
        ++m_fullQrStatistics.rebuilds;
        m_y.deviceCopyToAsync(m_qr);
        residual.deviceCopyToAsync(m_rhs);
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDgeqrf(handle(), m_rows, m_memory, m_qr.raw(), m_rows,
                m_tau.raw(), m_qrWorkspace->raw(), m_qrWorkspaceSize, m_info.raw()));
        } else {
            gpuErrChk(cusolverDnSgeqrf(handle(), m_rows, m_memory, m_qr.raw(), m_rows,
                m_tau.raw(), m_qrWorkspace->raw(), m_qrWorkspaceSize, m_info.raw()));
        }
        if (!solverSucceeded()) return false;
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDormqr(handle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                m_rows, 1, m_memory, m_qr.raw(), m_rows, m_tau.raw(), m_rhs.raw(), m_rows,
                m_qrWorkspace->raw(), m_qrWorkspaceSize, m_info.raw()));
        } else {
            gpuErrChk(cusolverDnSormqr(handle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                m_rows, 1, m_memory, m_qr.raw(), m_rows, m_tau.raw(), m_rhs.raw(), m_rows,
                m_qrWorkspace->raw(), m_qrWorkspaceSize, m_info.raw()));
        }
        if (!solverSucceeded()) return false;
        spock::detail::copy2DToHost(m_rHost.data(), m_memory * sizeof(T), m_qr.raw(), m_rows * sizeof(T),
            m_memory * sizeof(T), m_memory, spock::executionStream(m_y));
        spock::detail::copyToHost(m_rhsHost.data(), m_rhs.raw(), m_memory * sizeof(T), spock::executionStream(m_y));
        // GEQRF stores Householder vectors below R; they are not part of R.
        for (int col = 0; col < m_memory; ++col)
            for (int row = col + 1; row < m_memory; ++row) m_rHost[col * m_memory + row] = 0;
        return true;
    }

    bool selectColumns() {
        for (T value : m_rHost) if (!std::isfinite(value)) return false;
        for (T value : m_rhsHost) if (!std::isfinite(value)) return false;
        for (int col = 0; col < m_memory; ++col) {
            double norm = 0.;
            for (int row = 0; row < m_memory; ++row) norm = std::hypot(norm, m_rHost[col * m_memory + row]);
            m_fit.scale = std::max(m_fit.scale, norm);
        }
        if (!std::isfinite(m_fit.scale)) return false;
        if (m_fit.scale == 0) return true;
        std::fill(m_packed.begin(), m_packed.end(), T(0));
        for (int col = 0; col < m_memory; ++col) {
            for (int row = 0; row < m_memory; ++row)
                m_innovation[row] = double(m_rHost[col * m_memory + row]) / m_fit.scale;
            // Two-pass modified Gram--Schmidt makes the rank decision reliable
            // near dependence. Newer columns are considered first.
            for (int pass = 0; pass < 2; ++pass)
                for (size_t basis = 0; basis < m_fit.selectedColumns.size(); ++basis) {
                    double dot = 0.;
                    for (int row = 0; row < m_memory; ++row)
                        dot += m_basis[basis * m_memory + row] * m_innovation[row];
                    for (int row = 0; row < m_memory; ++row)
                        m_innovation[row] -= dot * m_basis[basis * m_memory + row];
                }
            double norm = 0.;
            for (double value : m_innovation) norm = std::hypot(norm, value);
            if (norm > m_options.rankTolerance) {
                const size_t selected = m_fit.selectedColumns.size();
                for (int row = 0; row < m_memory; ++row) {
                    m_basis[selected * m_memory + row] = m_innovation[row] / norm;
                    m_packed[selected * m_memory + row] = T(double(m_rHost[col * m_memory + row]) / m_fit.scale);
                }
                m_fit.selectedColumns.push_back(col);
            }
        }
        for (int row = 0; row < m_memory; ++row) {
            m_normalisedRhs[row] = double(m_rhsHost[row]) / m_fit.scale;
            if (!std::isfinite(m_normalisedRhs[row])) return false;
        }
        return true;
    }

    bool solveSelectedRidge() {
        spock::detail::copyToDevice(m_smallMatrix.raw(), m_packed.data(), m_packed.size() * sizeof(T), spock::executionStream(m_y));
        if constexpr (std::is_same_v<T, double>) {
            gpuErrChk(cusolverDnDgesvdj(handle(), CUSOLVER_EIG_MODE_VECTOR, 0, m_memory, m_memory,
                m_smallMatrix.raw(), m_memory, m_singularValues.raw(), m_u.raw(), m_memory, m_v.raw(), m_memory,
                m_svdWorkspace->raw(), m_svdWorkspaceSize, m_info.raw(), m_svdParameters.value));
        } else {
            gpuErrChk(cusolverDnSgesvdj(handle(), CUSOLVER_EIG_MODE_VECTOR, 0, m_memory, m_memory,
                m_smallMatrix.raw(), m_memory, m_singularValues.raw(), m_u.raw(), m_memory, m_v.raw(), m_memory,
                m_svdWorkspace->raw(), m_svdWorkspaceSize, m_info.raw(), m_svdParameters.value));
        }
        if (!solverSucceeded()) {
            m_fit.status = AndersonFitStatus::solverFailure;
            return false;
        }
        spock::detail::copyToHost(m_uHost.data(), m_u.raw(), m_uHost.size() * sizeof(T), spock::executionStream(m_y));
        spock::detail::copyToHost(m_vHost.data(), m_v.raw(), m_vHost.size() * sizeof(T), spock::executionStream(m_y));
        spock::detail::copyToHost(m_sHost.data(), m_singularValues.raw(), m_sHost.size() * sizeof(T), spock::executionStream(m_y));
        std::fill(m_solution.begin(), m_solution.end(), 0.);
        for (int mode = 0; mode < m_memory; ++mode) {
            const double sigma = m_sHost[mode];
            if (!std::isfinite(sigma) || sigma < 0) return false;
            double rhs = 0.;
            for (int row = 0; row < m_memory; ++row)
                rhs += double(m_uHost[mode * m_memory + row]) * m_normalisedRhs[row];
            const double weight = (sigma / (sigma * sigma + m_options.regularization)) * rhs;
            // gesvdj returns V (not V'). The appended zero columns have zero
            // applied coefficients; positive ridge also handles zero modes.
            for (size_t col = 0; col < m_fit.selectedColumns.size(); ++col)
                m_solution[col] += double(m_vHost[mode * m_memory + col]) * weight;
        }
        for (size_t j = 0; j < m_fit.selectedColumns.size(); ++j) {
            const T coefficient = T(m_solution[j]);
            if (!std::isfinite(coefficient)) return false;
            const size_t physical = m_fit.selectedColumns[j];
            m_gammaHost[physical] = coefficient;
            m_fit.coefficients[physical] = coefficient;
            m_fit.coefficientNorm = std::hypot(m_fit.coefficientNorm, double(coefficient));
        }
        return true;
    }

    const AndersonFitInfo &fallback(DTensor<T> &residual, DTensor<T> &direction, AndersonFitStatus status) {
        if (m_rollingQr) m_rollingQr->invalidate();
        m_fit.status = status;
        std::fill(m_fit.coefficients.begin(), m_fit.coefficients.end(), 0.);
        std::fill(m_gammaHost.begin(), m_gammaHost.end(), T(0));
        spock::detail::copyToDevice(m_gamma.raw(), m_gammaHost.data(), m_memory * sizeof(T), spock::executionStream(m_y));
        residual.deviceCopyToAsync(direction);
        direction *= T(-1);
        return m_fit;
    }

public:
    RegularisedAnderson(DTensor<T> &c, DTensor<T> &y, DTensor<T> &qr,
                       DTensor<T> &rhs, DTensor<T> &gamma, AndersonOptions options)
        : m_c(c), m_y(y), m_qr(qr), m_rhs(rhs), m_gamma(gamma),
          m_rows(validateWorkspaces(c, y, qr, rhs, gamma)), m_memory(int(y.numCols())),
          m_options(validatedOptions(options)),
          m_tau(m_memory), m_smallMatrix(m_memory, m_memory), m_u(m_memory, m_memory),
          m_v(m_memory, m_memory), m_singularValues(m_memory), m_info(1),
          m_rHost(m_memory * m_memory), m_rhsHost(m_memory), m_uHost(m_memory * m_memory),
          m_vHost(m_memory * m_memory), m_sHost(m_memory), m_packed(m_memory * m_memory), m_gammaHost(m_memory),
          m_basis(m_memory * m_memory), m_innovation(m_memory), m_normalisedRhs(m_memory), m_solution(m_memory) {
        spock::commonStream(c, y, qr, rhs, gamma);
        // Small solver buffers follow the history, including non-default streams.
        for (auto *tensor : {&m_tau, &m_smallMatrix, &m_u, &m_v, &m_singularValues})
            tensor->setStreamIdx(y.streamIdx());
        m_info.setStreamIdx(y.streamIdx());
        m_fit.selectedColumns.reserve(m_memory);
        m_fit.coefficients.resize(m_memory);
        allocateSolverWorkspaces();
        if (m_options.qr.incremental)
            m_rollingQr = std::make_unique<AndersonRollingQr<T>>(m_y, m_qr, m_rhs, m_options.qr, m_options.rankTolerance);
    }

    const AndersonOptions &options() const { return m_options; }
    void historyChanged() { if (m_rollingQr) m_rollingQr->historyChanged(); }
    void reset() {
        m_fullQrStatistics = {};
        if (m_rollingQr) m_rollingQr->reset();
    }
    const AndersonQrStatistics &qrStatistics() const {
        return m_rollingQr ? m_rollingQr->statistics() : m_fullQrStatistics;
    }

    const AndersonFitInfo &compute(DTensor<T> &residual, DTensor<T> &direction) {
        if (residual.numRows() != size_t(m_rows) || residual.numCols() != 1 || residual.numMats() != 1 ||
            direction.numRows() != size_t(m_rows) || direction.numCols() != 1 || direction.numMats() != 1)
            throw std::invalid_argument("Invalid Anderson residual/direction size");
        spock::commonStream(m_y, residual, direction);
        m_fit.selectedColumns.clear();
        std::fill(m_fit.coefficients.begin(), m_fit.coefficients.end(), 0.);
        std::fill(m_gammaHost.begin(), m_gammaHost.end(), T(0));
        m_fit.scale = m_fit.coefficientNorm = m_fit.directionRatio = 0.;
        m_fit.status = AndersonFitStatus::accepted;
        const double residualNorm = residual.normF();
        if (!std::isfinite(residualNorm)) throw std::runtime_error("Nonfinite current SPOCK residual");
        if (!std::isfinite(double(m_c.normF())) || !std::isfinite(double(m_y.normF())))
            return fallback(residual, direction, AndersonFitStatus::nonfiniteHistory);
        if (!compressHistory(residual)) return fallback(residual, direction, AndersonFitStatus::solverFailure);
        if (!selectColumns()) return fallback(residual, direction, AndersonFitStatus::nonfiniteHistory);
        if (m_fit.selectedColumns.empty()) return fallback(residual, direction, AndersonFitStatus::rankZero);
        if (!solveSelectedRidge()) return fallback(residual, direction,
            m_fit.status == AndersonFitStatus::solverFailure ? m_fit.status : AndersonFitStatus::nonfiniteSolution);
        if (m_fit.coefficientNorm > m_options.maxCoefficientNorm)
            return fallback(residual, direction, AndersonFitStatus::coefficientLimit);
        spock::detail::copyToDevice(m_gamma.raw(), m_gammaHost.data(), m_memory * sizeof(T), spock::executionStream(m_y));
        residual.deviceCopyToAsync(direction);
        direction *= T(-1);
        direction.addAB(m_c, m_gamma, T(-1), T(1));
        const double directionNorm = direction.normF();
        if (!std::isfinite(directionNorm)) return fallback(residual, direction, AndersonFitStatus::nonfiniteSolution);
        m_fit.directionRatio = residualNorm > 0 ? directionNorm / residualNorm : (directionNorm == 0 ? 0. : INFINITY);
        if (m_fit.directionRatio > m_options.maxDirectionRatio)
            return fallback(residual, direction, AndersonFitStatus::directionLimit);
        m_fit.status = AndersonFitStatus::accepted;
        return m_fit;
    }
};

#endif
