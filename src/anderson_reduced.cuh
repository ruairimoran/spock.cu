#ifndef SPOCK_ANDERSON_REDUCED_CUH
#define SPOCK_ANDERSON_REDUCED_CUH

#include "../include/gpu.cuh"
#include "anderson_fit_types.cuh"
#include <algorithm>
#include <memory>

namespace anderson_reduced_detail {

struct FitState {
    AndersonFitStatus status;
    int rank;
    double scale, coefficientNorm, directionRatio;
};

// Fixed header followed by memory selected indices and memory coefficients.
// One registered download retains the public host diagnostics without reading
// intermediate R, Q^T r, singular vectors, or singular values on the CPU.
enum DiagnosticField { statusField, rankField, scaleField, coefficientNormField,
                       directionRatioField, headerSize };

template<typename T>
__device__ double scaledLength(const T *values, int count) {
    double scale = 0.;
    for (int i = 0; i < count; ++i) scale = fmax(scale, fabs(double(values[i])));
    if (scale == 0.) return 0.;
    const double inverse = 1. / scale;
    double sum = 0.;
    for (int i = 0; i < count; ++i) {
        // The division path also handles subnormal scales whose reciprocal
        // overflows. Normal inputs require only one reciprocal per vector.
        const double value = isfinite(inverse) ? double(values[i]) * inverse : double(values[i]) / scale;
        sum += value * value;
    }
    return scale * sqrt(sum);
}

template<typename T>
__global__ void prepare(T *matrix, const T *rhs, int memory, double rankTolerance,
                        double *basis, double *innovation, double *normalisedRhs,
                        int *indices, T *gamma, FitState *state) {
    *state = {AndersonFitStatus::accepted, 0, 0., 0., 0.};
    // Cache deliberately aliases gamma with the first entries of its tall RHS
    // workspace. Preserve the projected RHS before clearing those coefficients.
    for (int i = 0; i < memory; ++i) normalisedRhs[i] = double(rhs[i]);
    for (int i = 0; i < memory; ++i) gamma[i] = T(0);
    for (int i = 0; i < memory * memory; ++i)
        if (!isfinite(double(matrix[i]))) state->status = AndersonFitStatus::nonfiniteHistory;
    for (int i = 0; i < memory; ++i)
        if (!isfinite(normalisedRhs[i])) state->status = AndersonFitStatus::nonfiniteHistory;
    if (state->status == AndersonFitStatus::accepted) {
        for (int col = 0; col < memory; ++col) {
            const double norm = scaledLength(matrix + col * memory, memory);
            state->scale = fmax(state->scale, norm);
        }
        if (!isfinite(state->scale)) state->status = AndersonFitStatus::nonfiniteHistory;
        else if (state->scale == 0.) state->status = AndersonFitStatus::rankZero;
    }
    if (state->status == AndersonFitStatus::accepted) {
        // Histories are short (normally three columns). One thread preserves
        // newest-first decisions and double intermediate arithmetic in both
        // precisions, matching the CPU reference's two-pass MGS filter.
        for (int col = 0; col < memory; ++col) {
            for (int row = 0; row < memory; ++row)
                innovation[row] = double(matrix[col * memory + row]) / state->scale;
            for (int pass = 0; pass < 2; ++pass)
                for (int selected = 0; selected < state->rank; ++selected) {
                    double dot = 0.;
                    for (int row = 0; row < memory; ++row)
                        dot += basis[selected * memory + row] * innovation[row];
                    for (int row = 0; row < memory; ++row)
                        innovation[row] -= dot * basis[selected * memory + row];
                }
            const double norm = scaledLength(innovation, memory);
            if (norm > rankTolerance) {
                const int selected = state->rank++;
                indices[selected] = col;
                for (int row = 0; row < memory; ++row) {
                    basis[selected * memory + row] = innovation[row] / norm;
                    // Compact towards earlier columns only: the source column
                    // and all unread later columns remain available in place.
                    matrix[selected * memory + row] = T(double(matrix[col * memory + row]) / state->scale);
                }
            }
        }
        for (int row = 0; row < memory; ++row) {
            normalisedRhs[row] /= state->scale;
            if (!isfinite(normalisedRhs[row])) state->status = AndersonFitStatus::nonfiniteHistory;
        }
        if (state->status == AndersonFitStatus::accepted && state->rank == 0)
            state->status = AndersonFitStatus::rankZero;
    }
    // Always give GPUtils a finite, fixed-size SVD input. Rejected preparation
    // uses a zero dummy problem; later kernels preserve its original reason.
    const int retained = state->status == AndersonFitStatus::accepted ? state->rank * memory : 0;
    for (int i = retained; i < memory * memory; ++i) matrix[i] = T(0);
}

template<typename T>
__global__ void coefficients(const T *u, const T *vt, const T *singularValues,
                             const int *solverInfo, const double *rhs, const int *indices,
                             int memory, double regularization, double maxNorm,
                             double *solution, T *gamma, FitState *state) {
    if (state->status != AndersonFitStatus::accepted) return;
    if (*solverInfo != 0) { state->status = AndersonFitStatus::solverFailure; return; }
    for (int col = 0; col < memory; ++col) solution[col] = 0.;
    for (int mode = 0; mode < memory; ++mode) {
        const double sigma = singularValues[mode];
        if (!isfinite(sigma) || sigma < 0.) {
            state->status = AndersonFitStatus::nonfiniteSolution;
            break;
        }
        double projection = 0.;
        for (int row = 0; row < memory; ++row) projection += double(u[mode * memory + row]) * rhs[row];
        const double weight = (sigma / (sigma * sigma + regularization)) * projection;
        for (int col = 0; col < state->rank; ++col)
            solution[col] += double(vt[col * memory + mode]) * weight;
    }
    if (state->status == AndersonFitStatus::accepted) {
        for (int col = 0; col < state->rank; ++col) {
            const T value = T(solution[col]);
            if (!isfinite(double(value))) {
                state->status = AndersonFitStatus::nonfiniteSolution;
                break;
            }
            gamma[indices[col]] = value;
        }
        state->coefficientNorm = scaledLength(gamma, memory);
        if (state->status == AndersonFitStatus::accepted && state->coefficientNorm > maxNorm)
            state->status = AndersonFitStatus::coefficientLimit;
    }
    if (state->status != AndersonFitStatus::accepted)
        for (int col = 0; col < memory; ++col) gamma[col] = T(0);
}

// GPUtils normF() returns a host scalar. This device-output scaled reduction
// avoids that completion boundary and avoids squaring extreme raw values.
struct ScaledNorm { double scale, sum; };

__device__ inline ScaledNorm combine(ScaledNorm a, ScaledNorm b) {
    if (isnan(a.scale) || isnan(b.scale)) return {NAN, 0.};
    if (isinf(a.scale) || isinf(b.scale)) return {INFINITY, 0.};
    if (a.scale < b.scale) { const auto temporary = a; a = b; b = temporary; }
    if (a.scale == 0.) return {0., 0.};
    const double ratio = b.scale / a.scale;
    return {a.scale, a.sum + b.sum * ratio * ratio};
}

template<typename T>
__global__ void directionNorm(const T *direction, size_t rows, const FitState *state, ScaledNorm *partials) {
    __shared__ double shared[256];
    if (state->status != AndersonFitStatus::accepted) {
        if (threadIdx.x == 0) partials[blockIdx.x] = {0., 0.};
        return;
    }
    double scale = 0.;
    for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += size_t(blockDim.x) * gridDim.x) {
        const double value = fabs(double(direction[row]));
        scale = isfinite(value) ? fmax(scale,value) : INFINITY;
    }
    shared[threadIdx.x] = scale;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmax(shared[threadIdx.x],shared[threadIdx.x + stride]);
        __syncthreads();
    }
    scale = shared[0];
    if (scale == 0. || !isfinite(scale)) {
        if (threadIdx.x == 0) partials[blockIdx.x] = {scale,0.};
        return;
    }
    const double inverse = 1. / scale;
    double sum = 0.;
    for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += size_t(blockDim.x) * gridDim.x) {
        const double value = isfinite(inverse) ? double(direction[row])*inverse : double(direction[row])/scale;
        sum += value*value;
    }
    __syncthreads(); // All threads have read the shared block scale.
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = {scale,shared[0]};
}

template<typename T>
__global__ void finish(const ScaledNorm *partials, int blocks, double residualNorm, double maxRatio,
                       int memory, const int *indices, T *gamma, FitState *state, double *diagnostics) {
    __shared__ ScaledNorm shared[256];
    shared[threadIdx.x] = threadIdx.x < blocks ? partials[threadIdx.x] : ScaledNorm{0.,0.};
    __syncthreads();
    // A serial merge is expensive even for 256 partials on this GPU. Merge the
    // scaled sums in parallel, then let one thread apply the small decisions.
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] = combine(shared[threadIdx.x],shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    if (state->status == AndersonFitStatus::accepted) {
        const ScaledNorm norm = shared[0];
        // normF() returns T in the reference, including overflow in float.
        const double length = T(norm.scale * sqrt(norm.sum));
        if (!isfinite(length)) state->status = AndersonFitStatus::nonfiniteSolution;
        else {
            state->directionRatio = residualNorm > 0. ? length / residualNorm : (length == 0. ? 0. : INFINITY);
            if (state->directionRatio > maxRatio) state->status = AndersonFitStatus::directionLimit;
        }
    }
    if (state->status != AndersonFitStatus::accepted)
        for (int col = 0; col < memory; ++col) gamma[col] = T(0);
    diagnostics[statusField] = int(state->status);
    diagnostics[rankField] = state->rank;
    diagnostics[scaleField] = state->scale;
    diagnostics[coefficientNormField] = state->coefficientNorm;
    diagnostics[directionRatioField] = state->directionRatio;
    for (int col = 0; col < memory; ++col) {
        diagnostics[headerSize + col] = col < state->rank ? indices[col] : -1;
        diagnostics[headerSize + memory + col] = double(gamma[col]);
    }
}

template<typename T>
__global__ void loadCpuFit(const double *diagnostics, int memory, int *indices, T *gamma, FitState *state) {
    *state = {AndersonFitStatus(int(diagnostics[statusField])), int(diagnostics[rankField]),
              diagnostics[scaleField], diagnostics[coefficientNormField], 0.};
    for (int col = 0; col < memory; ++col) {
        indices[col] = int(diagnostics[headerSize + col]);
        gamma[col] = T(diagnostics[headerSize + memory + col]);
    }
}

template<typename T>
__global__ void fallback(const T *residual, T *direction, size_t rows, T *gamma, int memory,
                         const FitState *state = nullptr) {
    if (state && state->status == AndersonFitStatus::accepted) return;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += size_t(blockDim.x) * gridDim.x) direction[i] = -residual[i];
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(memory); i += size_t(blockDim.x) * gridDim.x) gamma[i] = T(0);
}

} // namespace anderson_reduced_detail

template<typename T>
class AndersonDeviceFit {
    int m_memory, m_normBlocks;
    size_t m_rows;
    cudaStream_t m_stream;
    DTensor<T> &m_gamma;
    DTensor<T> m_matrix;
    DTensor<double> m_basis, m_innovation, m_rhs, m_solution, m_diagnostics;
    DTensor<int> m_indices;
    DTensor<anderson_reduced_detail::FitState> m_state;
    DTensor<anderson_reduced_detail::ScaledNorm> m_normPartials;
    RegisteredHostVector<double> m_hostDiagnostics;
    std::unique_ptr<Svd<T>> m_svd;

public:
    AndersonDeviceFit(int memory, size_t rows, DTensor<T> &gamma, bool deviceSvd = true)
        : m_memory(memory), m_normBlocks(int(std::min(size_t(256), numBlocks(rows, 256)))),
          m_rows(rows), m_stream(spock::executionStream(gamma)), m_gamma(gamma),
          m_matrix(memory, memory), m_basis(memory, memory), m_innovation(memory), m_rhs(memory),
          m_solution(memory), m_diagnostics(anderson_reduced_detail::headerSize + 2 * memory),
          m_indices(memory), m_state(1), m_normPartials(m_normBlocks),
          m_hostDiagnostics(m_diagnostics.numEl()) {
        m_matrix.setStreamIdx(gamma.streamIdx());
        for (auto *tensor : {&m_basis, &m_innovation, &m_rhs, &m_solution, &m_diagnostics})
            tensor->setStreamIdx(gamma.streamIdx());
        m_indices.setStreamIdx(gamma.streamIdx());
        m_state.setStreamIdx(gamma.streamIdx());
        m_normPartials.setStreamIdx(gamma.streamIdx());
        if (deviceSvd) m_svd = std::make_unique<Svd<T>>(m_matrix, true, true);
    }

    DTensor<T> &matrix() { return m_matrix; }

    // Both backends use the same GPU direction construction, norm check and
    // fallback. CPU coefficients and metadata cross the bus in one upload.
    void publishCpuFit(const AndersonFitInfo &info) {
        using namespace anderson_reduced_detail;
        m_hostDiagnostics[statusField] = int(info.status);
        m_hostDiagnostics[rankField] = info.selectedColumns.size();
        m_hostDiagnostics[scaleField] = info.scale;
        m_hostDiagnostics[coefficientNormField] = info.coefficientNorm;
        m_hostDiagnostics[directionRatioField] = 0.;
        for (int col = 0; col < m_memory; ++col) {
            m_hostDiagnostics[headerSize + col] = size_t(col) < info.selectedColumns.size() ? info.selectedColumns[col] : 0;
            m_hostDiagnostics[headerSize + m_memory + col] = info.coefficients[col];
        }
        m_diagnostics.uploadAsync(m_hostDiagnostics);
        loadCpuFit<<<1,1,0,m_stream>>>(m_diagnostics.raw(),m_memory,m_indices.raw(),m_gamma.raw(),m_state.raw());
        gpuErrChk(cudaGetLastError());
        // finish() orders its download after this upload on the same stream;
        // the registered host buffer stays unchanged until that completion.
    }

    void solve(const DTensor<T> &rhs, double rankTolerance, double regularization, double maxCoefficientNorm) {
        spock::commonStream(m_matrix, rhs);
        anderson_reduced_detail::prepare<<<1, 1, 0, m_stream>>>(m_matrix.raw(), rhs.raw(), m_memory,
            rankTolerance, m_basis.raw(), m_innovation.raw(), m_rhs.raw(), m_indices.raw(), m_gamma.raw(), m_state.raw());
        gpuErrChk(cudaGetLastError());
        m_svd->factorise();
        anderson_reduced_detail::coefficients<<<1, 1, 0, m_stream>>>(m_svd->leftSingularVectors().value()->raw(),
            m_svd->rightSingularVectors().raw(), m_svd->singularValues().raw(), m_svd->info().raw(),
            m_rhs.raw(), m_indices.raw(), m_memory, regularization, maxCoefficientNorm,
            m_solution.raw(), m_gamma.raw(), m_state.raw());
        gpuErrChk(cudaGetLastError());
    }

    void finish(const DTensor<T> &residual, DTensor<T> &direction, double residualNorm, double maxDirectionRatio,
                AndersonFitInfo &info) {
        spock::commonStream(m_matrix, residual, direction);
        anderson_reduced_detail::directionNorm<<<m_normBlocks, 256, 0, m_stream>>>(
            direction.raw(), m_rows, m_state.raw(), m_normPartials.raw());
        gpuErrChk(cudaGetLastError());
        anderson_reduced_detail::finish<T><<<1, 256, 0, m_stream>>>(m_normPartials.raw(), m_normBlocks,
            residualNorm, maxDirectionRatio, m_memory, m_indices.raw(), m_gamma.raw(), m_state.raw(), m_diagnostics.raw());
        gpuErrChk(cudaGetLastError());
        anderson_reduced_detail::fallback<<<m_normBlocks, 256, 0, m_stream>>>(
            residual.raw(), direction.raw(), m_rows, m_gamma.raw(), m_memory, m_state.raw());
        gpuErrChk(cudaGetLastError());
        m_diagnostics.downloadAsync(m_hostDiagnostics);
        gpuErrChk(cudaStreamSynchronize(m_stream));
        using namespace anderson_reduced_detail;
        info.status = AndersonFitStatus(int(m_hostDiagnostics[statusField]));
        info.scale = m_hostDiagnostics[scaleField];
        info.coefficientNorm = m_hostDiagnostics[coefficientNormField];
        info.directionRatio = m_hostDiagnostics[directionRatioField];
        info.selectedColumns.clear();
        const int rank = int(m_hostDiagnostics[rankField]);
        for (int col = 0; col < rank; ++col) info.selectedColumns.push_back(size_t(m_hostDiagnostics[headerSize + col]));
        for (int col = 0; col < m_memory; ++col) info.coefficients[col] = m_hostDiagnostics[headerSize + m_memory + col];
    }
};

#endif
