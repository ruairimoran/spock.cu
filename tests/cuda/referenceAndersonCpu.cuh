#pragma once
// Frozen pre-change reduced fit: numerical regression reference only.
#include "anderson.cuh"

// A type-II Anderson fit with borrowed tall workspaces. All device storage is
// allocated at construction; compute() transfers only O(memory^2) fit data
// between host and device. C and Y must contain matching, newest-first columns,
// with C = delta(iterate) - delta(residual).
template<typename T>
class CpuReferenceAnderson {
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
    DTensor<T> m_smallMatrix;
    std::unique_ptr<QRFactoriser<T>> m_fullQr;
    std::unique_ptr<Svd<T>> m_svd;
    std::vector<T> m_rHost, m_rhsHost, m_uHost, m_vtHost, m_sHost, m_packed, m_gammaHost;
    std::vector<double> m_basis, m_innovation, m_normalisedRhs, m_solution;
    AndersonFitInfo m_fit;
    std::unique_ptr<AndersonRollingQr<T>> m_rollingQr;
    AndersonQrStatistics m_fullQrStatistics;

    bool solverSucceeded(const DTensor<int> &deviceInfo) {
        int info;
        spock::detail::copyToHost(&info, deviceInfo.raw(), sizeof(info), spock::executionStream(deviceInfo));
        return info == 0;
    }

    bool compressHistory(DTensor<T> &residual) {
        if (m_rollingQr) return m_rollingQr->compress(residual, m_rHost, m_rhsHost);
        ++m_fullQrStatistics.rebuilds;
        m_y.deviceCopyToAsync(m_qr);
        residual.deviceCopyToAsync(m_rhs);
        m_fullQr->factorise();
        if (!solverSucceeded(m_fullQr->info())) return false;
        m_fullQr->applyQt(m_rhs);
        if (!solverSucceeded(m_fullQr->info())) return false;
        // Reuse the small SVD input for R. The ridge fit refills this matrix
        // after rank filtering, so no additional R storage is needed.
        m_fullQr->extractR(m_smallMatrix);
        const auto stream = spock::executionStream(m_y);
        spock::detail::copyToHost(m_rHost.data(), m_smallMatrix.raw(), m_rHost.size() * sizeof(T), stream);
        spock::detail::copyToHost(m_rhsHost.data(), m_rhs.raw(), m_memory * sizeof(T), stream);
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
        m_svd->factorise();
        if (!solverSucceeded(m_svd->info())) {
            m_fit.status = AndersonFitStatus::solverFailure;
            return false;
        }
        const auto &u = *m_svd->leftSingularVectors().value();
        const auto &vt = m_svd->rightSingularVectors();
        const auto &singularValues = m_svd->singularValues();
        const auto stream = spock::executionStream(m_y);
        spock::detail::copyToHost(m_uHost.data(), u.raw(), m_uHost.size() * sizeof(T), stream);
        spock::detail::copyToHost(m_vtHost.data(), vt.raw(), m_vtHost.size() * sizeof(T), stream);
        spock::detail::copyToHost(m_sHost.data(), singularValues.raw(), m_sHost.size() * sizeof(T), stream);
        std::fill(m_solution.begin(), m_solution.end(), 0.);
        for (int mode = 0; mode < m_memory; ++mode) {
            const double sigma = m_sHost[mode];
            if (!std::isfinite(sigma) || sigma < 0) return false;
            double rhs = 0.;
            for (int row = 0; row < m_memory; ++row)
                rhs += double(m_uHost[mode * m_memory + row]) * m_normalisedRhs[row];
            const double weight = (sigma / (sigma * sigma + m_options.regularization)) * rhs;
            // GPUtils returns V^T in column-major storage, so V(col, mode)
            // is V^T(mode, col). Zero-padded columns retain zero coefficients.
            for (size_t col = 0; col < m_fit.selectedColumns.size(); ++col)
                m_solution[col] += double(m_vtHost[col * m_memory + mode]) * weight;
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
    CpuReferenceAnderson(DTensor<T> &c, DTensor<T> &y, DTensor<T> &qr,
                       DTensor<T> &rhs, DTensor<T> &gamma, AndersonOptions options)
        : m_c(c), m_y(y), m_qr(qr), m_rhs(rhs), m_gamma(gamma),
          m_rows(validateWorkspaces(c, y, qr, rhs, gamma)), m_memory(int(y.numCols())),
          m_options(validatedOptions(options)),
          m_smallMatrix(m_memory, m_memory),
          m_rHost(m_memory * m_memory), m_rhsHost(m_memory), m_uHost(m_memory * m_memory),
          m_vtHost(m_memory * m_memory), m_sHost(m_memory), m_packed(m_memory * m_memory), m_gammaHost(m_memory),
          m_basis(m_memory * m_memory), m_innovation(m_memory), m_normalisedRhs(m_memory), m_solution(m_memory) {
        spock::commonStream(c, y, qr, rhs, gamma);
        // Small solver buffers follow the history, including non-default streams.
        m_smallMatrix.setStreamIdx(y.streamIdx());
        m_fit.selectedColumns.reserve(m_memory);
        m_fit.coefficients.resize(m_memory);
        // Construct once on the history stream. factorise() overwrites this
        // input; solveSelectedRidge() refills it before every fit.
        m_svd = std::make_unique<Svd<T>>(m_smallMatrix, true, true);
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

