#ifndef SPOCK_ANDERSON_CUH
#define SPOCK_ANDERSON_CUH

#include "../include/gpu.cuh"
#include "anderson_qr.cuh"
#include "anderson_reduced.cuh"
#include "anderson_cpu.cuh"
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
    AndersonFitBackend fitBackend = AndersonFitBackend::gpu;

    void validate() const {
        qr.validate();
        if (fitBackend != AndersonFitBackend::gpu && fitBackend != AndersonFitBackend::cpu)
            throw std::invalid_argument("Invalid Anderson reduced-fit backend");
        if (!std::isfinite(regularization) || regularization <= 0 ||
            !std::isfinite(rankTolerance) || rankTolerance < 0 || rankTolerance >= 1 ||
            !std::isfinite(maxCoefficientNorm) || maxCoefficientNorm <= 0 ||
            !std::isfinite(maxDirectionRatio) || maxDirectionRatio <= 0)
            throw std::invalid_argument("Invalid regularised Anderson options");
    }
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
// allocated at construction. Reduced fitting can use the GPU or CPU backend;
// GPU fitting transfers final diagnostics, while CPU fitting transfers the
// compressed data and coefficients. C and Y must have matching, newest-first columns,
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

    DTensor<T> &m_c, &m_y, &m_qr, &m_rhs, &m_gamma;
    int m_rows, m_memory;
    AndersonOptions m_options;
    std::unique_ptr<AndersonDeviceFit<T>> m_reduced;
    std::unique_ptr<AndersonCpuFit<T>> m_cpu;
    std::unique_ptr<RegisteredHostVector<T>> m_cpuR, m_cpuRhs;
    std::unique_ptr<DTensor<T>> m_rhsPrefix;
    std::unique_ptr<QRFactoriser<T>> m_fullQr;
    AndersonFitInfo m_fit;
    std::unique_ptr<AndersonRollingQr<T>> m_rollingQr;
    AndersonQrStatistics m_fullQrStatistics;

    bool solverSucceeded(const DTensor<int> &deviceInfo) {
        int info;
        spock::detail::copyToHost(&info, deviceInfo.raw(), sizeof(info), spock::executionStream(deviceInfo));
        return info == 0;
    }

    bool compressHistory(DTensor<T> &residual) {
        if (m_rollingQr) return m_cpu
            ? m_rollingQr->compressHost(residual,*m_cpuR,*m_cpuRhs)
            : m_rollingQr->compressDevice(residual,m_reduced->matrix());
        ++m_fullQrStatistics.rebuilds;
        m_y.deviceCopyToAsync(m_qr);
        residual.deviceCopyToAsync(m_rhs);
        m_fullQr->factorise();
        if (!solverSucceeded(m_fullQr->info())) return false;
        m_fullQr->applyQt(m_rhs);
        if (!solverSucceeded(m_fullQr->info())) return false;
        m_fullQr->extractR(m_reduced->matrix());
        if (m_cpu) {
            m_reduced->matrix().downloadAsync(*m_cpuR);
            m_rhsPrefix->downloadAsync(*m_cpuRhs);
            gpuErrChk(cudaStreamSynchronize(spock::executionStream(m_y)));
        }
        return true;
    }

    const AndersonFitInfo &fallback(DTensor<T> &residual, DTensor<T> &direction, AndersonFitStatus status) {
        if (m_rollingQr) m_rollingQr->invalidate();
        m_fit.status = status;
        std::fill(m_fit.coefficients.begin(), m_fit.coefficients.end(), 0.);
        const auto blocks = unsigned(std::min(size_t(256), numBlocks(size_t(m_rows), 256)));
        anderson_reduced_detail::fallback<<<blocks, 256, 0, spock::executionStream(m_y)>>>(
            residual.raw(), direction.raw(), size_t(m_rows), m_gamma.raw(), m_memory);
        gpuErrChk(cudaGetLastError());
        return m_fit;
    }

public:
    RegularisedAnderson(DTensor<T> &c, DTensor<T> &y, DTensor<T> &qr,
                       DTensor<T> &rhs, DTensor<T> &gamma, AndersonOptions options)
        : m_c(c), m_y(y), m_qr(qr), m_rhs(rhs), m_gamma(gamma),
          m_rows(validateWorkspaces(c, y, qr, rhs, gamma)), m_memory(int(y.numCols())),
          m_options(validatedOptions(options)) {
        spock::commonStream(c, y, qr, rhs, gamma);
        const bool cpu = m_options.fitBackend == AndersonFitBackend::cpu;
        m_reduced = std::make_unique<AndersonDeviceFit<T>>(m_memory,m_rows,m_gamma,!cpu);
        if (cpu) {
            m_cpu = std::make_unique<AndersonCpuFit<T>>(m_memory);
            m_cpuR = std::make_unique<RegisteredHostVector<T>>(m_memory*m_memory);
            m_cpuRhs = std::make_unique<RegisteredHostVector<T>>(m_memory);
            m_rhsPrefix = std::make_unique<DTensor<T>>(m_rhs,0,0,m_memory-1);
            m_rhsPrefix->setStreamIdx(m_y.streamIdx());
        }
        m_fit.selectedColumns.reserve(m_memory);
        m_fit.coefficients.resize(m_memory);
        if (m_options.qr.incremental)
            m_rollingQr = std::make_unique<AndersonRollingQr<T>>(m_y, m_qr, m_rhs, m_options.qr, m_options.rankTolerance);
        else
            m_fullQr = std::make_unique<QRFactoriser<T>>(m_qr);
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
        m_fit.scale = m_fit.coefficientNorm = m_fit.directionRatio = 0.;
        m_fit.status = AndersonFitStatus::accepted;
        const double residualNorm = residual.normF();
        if (!std::isfinite(residualNorm)) throw std::runtime_error("Nonfinite current SPOCK residual");
        if (!std::isfinite(double(m_c.normF())) || !std::isfinite(double(m_y.normF())))
            return fallback(residual, direction, AndersonFitStatus::nonfiniteHistory);
        if (!compressHistory(residual)) return fallback(residual, direction, AndersonFitStatus::solverFailure);
        if (m_cpu) {
            m_cpu->compute(m_cpuR->data(),m_cpuRhs->data(),m_options.rankTolerance,
                m_options.regularization,m_options.maxCoefficientNorm,m_fit);
            m_reduced->publishCpuFit(m_fit);
        } else {
            m_reduced->solve(m_rhs,m_options.rankTolerance,m_options.regularization,m_options.maxCoefficientNorm);
        }
        residual.deviceCopyToAsync(direction);
        direction *= T(-1);
        direction.addAB(m_c, m_gamma, T(-1), T(1));
        m_reduced->finish(residual, direction, residualNorm, m_options.maxDirectionRatio, m_fit);
        if (!m_fit.accepted() && m_rollingQr) m_rollingQr->invalidate();
        return m_fit;
    }
};

#endif
