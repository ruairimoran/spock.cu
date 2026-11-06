#ifndef SPOCK_ANDERSON_CPU_CUH
#define SPOCK_ANDERSON_CPU_CUH

#include "anderson_fit_types.cuh"
#include "cpu_svd.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Only the reduced memory-by-memory problem is processed here. Tall history,
// QR compression and direction formation remain on the GPU. All host scratch
// and SVD workspaces are allocated by construction, including deficient fits.
template<typename T>
class AndersonCpuFit {
    int m_memory;
    CpuSvd<T> m_svd;
    std::vector<T> m_packed;
    std::vector<double> m_basis, m_innovation, m_rhs, m_solution;

    static void reject(AndersonFitInfo &fit, AndersonFitStatus status) {
        fit.status = status;
        std::fill(fit.coefficients.begin(),fit.coefficients.end(),0.);
    }

public:
    explicit AndersonCpuFit(int memory)
        : m_memory(memory), m_svd(memory), m_packed(memory*memory), m_basis(memory*memory),
          m_innovation(memory), m_rhs(memory), m_solution(memory) {}

    void compute(const T *r, const T *rhs, double rankTolerance, double regularization,
                 double maxCoefficientNorm, AndersonFitInfo &fit) {
        if (fit.coefficients.size() != size_t(m_memory))
            throw std::invalid_argument("CPU Anderson diagnostics have the wrong size");
        fit.selectedColumns.clear();
        std::fill(fit.coefficients.begin(),fit.coefficients.end(),0.);
        fit.scale = fit.coefficientNorm = fit.directionRatio = 0.;
        fit.status = AndersonFitStatus::accepted;
        for (int i = 0; i < m_memory*m_memory; ++i)
            if (!std::isfinite(r[i])) { reject(fit,AndersonFitStatus::nonfiniteHistory); return; }
        for (int i = 0; i < m_memory; ++i)
            if (!std::isfinite(rhs[i])) { reject(fit,AndersonFitStatus::nonfiniteHistory); return; }
        for (int col = 0; col < m_memory; ++col) {
            double norm = 0.;
            for (int row = 0; row < m_memory; ++row) norm = std::hypot(norm,double(r[col*m_memory+row]));
            fit.scale = std::max(fit.scale,norm);
        }
        if (!std::isfinite(fit.scale)) { reject(fit,AndersonFitStatus::nonfiniteHistory); return; }
        if (fit.scale == 0.) { reject(fit,AndersonFitStatus::rankZero); return; }
        std::fill(m_packed.begin(),m_packed.end(),T(0));
        for (int col = 0; col < m_memory; ++col) {
            for (int row = 0; row < m_memory; ++row) m_innovation[row] = double(r[col*m_memory+row])/fit.scale;
            // Retain the same ordered, two-pass MGS filter as the GPU backend.
            for (int pass = 0; pass < 2; ++pass)
                for (size_t selected = 0; selected < fit.selectedColumns.size(); ++selected) {
                    double dot = 0.;
                    for (int row = 0; row < m_memory; ++row) dot += m_basis[selected*m_memory+row]*m_innovation[row];
                    for (int row = 0; row < m_memory; ++row) m_innovation[row] -= dot*m_basis[selected*m_memory+row];
                }
            double norm = 0.;
            for (double value : m_innovation) norm = std::hypot(norm,value);
            if (norm > rankTolerance) {
                const auto selected = fit.selectedColumns.size();
                for (int row = 0; row < m_memory; ++row) {
                    m_basis[selected*m_memory+row] = m_innovation[row]/norm;
                    m_packed[selected*m_memory+row] = T(double(r[col*m_memory+row])/fit.scale);
                }
                fit.selectedColumns.push_back(col);
            }
        }
        for (int row = 0; row < m_memory; ++row) {
            m_rhs[row] = double(rhs[row])/fit.scale;
            if (!std::isfinite(m_rhs[row])) { reject(fit,AndersonFitStatus::nonfiniteHistory); return; }
        }
        if (fit.selectedColumns.empty()) { reject(fit,AndersonFitStatus::rankZero); return; }
        if (!m_svd.factorise(m_packed.data())) { reject(fit,AndersonFitStatus::solverFailure); return; }
        std::fill(m_solution.begin(),m_solution.end(),0.);
        const auto u = m_svd.left(), v = m_svd.right(), singularValues = m_svd.singularValues();
        for (int mode = 0; mode < m_memory; ++mode) {
            const double sigma = singularValues[mode];
            if (!std::isfinite(sigma) || sigma < 0.) { reject(fit,AndersonFitStatus::nonfiniteSolution); return; }
            double projection = 0.;
            for (int row = 0; row < m_memory; ++row) projection += double(u[mode*m_memory+row])*m_rhs[row];
            const double weight = (sigma/(sigma*sigma+regularization))*projection;
            // Eigen returns column-major V, whereas GPUtils returns V^T.
            for (size_t col = 0; col < fit.selectedColumns.size(); ++col)
                m_solution[col] += double(v[mode*m_memory+col])*weight;
        }
        for (size_t col = 0; col < fit.selectedColumns.size(); ++col) {
            const T coefficient = T(m_solution[col]);
            if (!std::isfinite(coefficient)) { reject(fit,AndersonFitStatus::nonfiniteSolution); return; }
            fit.coefficients[fit.selectedColumns[col]] = coefficient;
            fit.coefficientNorm = std::hypot(fit.coefficientNorm,double(coefficient));
        }
        if (fit.coefficientNorm > maxCoefficientNorm) reject(fit,AndersonFitStatus::coefficientLimit);
    }
};

#endif
