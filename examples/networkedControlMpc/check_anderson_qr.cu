#include <anderson.cuh>
#include <cuda_profiler_api.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

using Writer = rapidjson::Writer<rapidjson::StringBuffer>;

static void writeQrStatistics(Writer &out, const AndersonQrStatistics &s) {
    out.StartObject();
    out.Key("rebuilds"); out.Uint64(s.rebuilds);
    out.Key("updates"); out.Uint64(s.updates);
    out.Key("reuses"); out.Uint64(s.reuses);
    out.Key("update_fallbacks"); out.Uint64(s.updateFallbacks);
    out.Key("orthogonality_checks"); out.Uint64(s.orthogonalityChecks);
    out.Key("last_orthogonality_error");
    if (std::isfinite(s.lastOrthogonalityError)) out.Double(s.lastOrthogonalityError); else out.Null();
    out.EndObject();
}

template<typename T>
void checkSequence(const rapidjson::Document &input) {
    const auto &frames = input["frames"];
    const size_t rows = frames[0]["residual"].Size(), memory = frames[0]["Y"][0].Size();
    std::vector<T> c(rows * memory), y(rows * memory), residual(rows), direction(rows), basis(rows * memory);
    DTensor<T> dC(rows, memory), dY(rows, memory), dResidual(rows), qr(rows, memory), rhs(rows), gamma(memory), dDirection(rows);
    AndersonOptions options;
    options.enabled = options.qr.incremental = true;
    options.rankTolerance = input["rank_tolerance"].GetDouble();
    options.qr.rebuildInterval = input["rebuild_interval"].GetUint64();
    options.qr.checkInterval = input["check_interval"].GetUint64();
    if (input.HasMember("orthogonality_tolerance"))
        options.qr.orthogonalityTolerance = input["orthogonality_tolerance"].GetDouble();
    if (input.HasMember("max_coefficient_norm")) options.maxCoefficientNorm = input["max_coefficient_norm"].GetDouble();
    RegularisedAnderson<T> fit(dC, dY, qr, rhs, gamma, options);
    const size_t allocated = Session::getInstance().totalAllocatedBytes();
    rapidjson::StringBuffer buffer; Writer out(buffer);
    out.StartObject(); out.Key("name"); out.String(input["name"].GetString());
    out.Key("frames"); out.StartArray();
    for (const auto &frame : frames.GetArray()) {
        for (size_t row = 0; row < rows; ++row) {
            residual[row] = T(frame["residual"][row].GetDouble());
            for (size_t col = 0; col < memory; ++col) {
                c[col * rows + row] = T(frame["C"][row][col].GetDouble());
                y[col * rows + row] = T(frame["Y"][row][col].GetDouble());
            }
        }
        dC.upload(c); dY.upload(y); dResidual.upload(residual);
        if (frame.HasMember("reset") && frame["reset"].GetBool()) fit.reset();
        if (frame.HasMember("damage_basis") && frame["damage_basis"].GetBool()) qr *= T(1.1);
        const size_t changes = frame.HasMember("history_changes") ? frame["history_changes"].GetUint64() : 1;
        for (size_t j = 0; j < changes; ++j) fit.historyChanged();
        const auto &info = fit.compute(dResidual, dDirection);
        dDirection.download(direction); qr.download(basis);
        if (Session::getInstance().totalAllocatedBytes() != allocated)
            throw std::runtime_error("Rolling QR changed its device allocation");
        out.StartObject();
        out.Key("status"); out.String(andersonStatusName(info.status));
        out.Key("accepted_indices"); out.StartArray();
        for (size_t index : info.selectedColumns) out.Uint64(index);
        out.EndArray();
        out.Key("gamma"); out.StartArray(); for (double value : info.coefficients) out.Double(value); out.EndArray();
        out.Key("direction"); out.StartArray(); for (T value : direction) out.Double(value); out.EndArray();
        out.Key("Q_columns"); out.StartArray(); for (T value : basis) out.Double(value); out.EndArray();
        out.Key("qr"); writeQrStatistics(out, fit.qrStatistics());
        out.EndObject();
    }
    out.EndArray(); out.EndObject(); std::cout << buffer.GetString() << '\n';
}

__device__ double fixtureValue(uint32_t row, uint32_t step) {
    uint32_t value = row * 747796405u + step * 2891336453u + 277803737u;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    value = (value >> 22u) ^ value;
    return double(value) / 2147483648. - 1.;
}

__global__ void generateWindow(double *c, double *y, double *residual, int rows, int memory, int step) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int col = 0; col < memory; ++col) {
        const double value = fixtureValue(row, step + memory - col);
        y[col * rows + row] = value;
        c[col * rows + row] = 0.2 * value + 0.01 * fixtureValue(row, step + memory - col + 1031);
    }
    residual[row] = 0.7 * y[row] + 0.2 * y[(memory - 1) * rows + row] + 0.3 * fixtureValue(row, step + 7411);
}

// Both paths see the same rolling histories and residuals. CUDA intervals
// include the complete Anderson fit; history generation and comparisons are
// outside the intervals. No changed convergence trajectory can bias this test.
void benchmark(int rows, int steps) {
    if (rows < 3 || steps < 1 || rows > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument("Invalid benchmark dimensions");
    const int memory = 3, warmup = 10;
    DTensor<double> c(rows, memory), y(rows, memory), residual(rows);
    DTensor<double> fullQ(rows, memory), updateQ(rows, memory), fullWork(rows), updateWork(rows);
    DTensor<double> fullGamma(memory), updateGamma(memory), fullDirection(rows), updateDirection(rows), difference(rows);
    AndersonOptions options; options.enabled = true;
    RegularisedAnderson<double> full(c, y, fullQ, fullWork, fullGamma, options);
    options.qr.incremental = true;
    RegularisedAnderson<double> update(c, y, updateQ, updateWork, updateGamma, options);
    std::vector<double> fullTimes, updateTimes;
    fullTimes.reserve(steps); updateTimes.reserve(steps);
    std::vector<double> a(memory), b(memory);
    const size_t allocated = Session::getInstance().totalAllocatedBytes();
    double maxDirectionError = 0., maxCoefficientError = 0.;
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start)); gpuErrChk(cudaEventCreate(&stop));
    cudaStream_t stream; gpuErrChk(cublasGetStream(Session::getInstance().cuBlasHandle(), &stream));
    const bool profile = std::getenv("SPOCK_PROFILE_RESIDENT") != nullptr;
    for (int step = 0; step < steps + warmup; ++step) {
        generateWindow<<<(rows + 255) / 256, 256, 0, stream>>>(c.raw(), y.raw(), residual.raw(), rows, memory, step);
        gpuErrChk(cudaGetLastError()); gpuErrChk(cudaStreamSynchronize(stream));
        full.historyChanged(); update.historyChanged();
        if (profile && step == warmup) gpuErrChk(cudaProfilerStart());
        auto run = [&](RegularisedAnderson<double> &fit, DTensor<double> &direction, std::vector<double> &times) {
            gpuErrChk(cudaEventRecord(start, stream));
            if (!fit.compute(residual, direction).accepted()) throw std::runtime_error("Benchmark direction rejected");
            gpuErrChk(cudaEventRecord(stop, stream)); gpuErrChk(cudaEventSynchronize(stop));
            float milliseconds; gpuErrChk(cudaEventElapsedTime(&milliseconds, start, stop));
            if (step >= warmup) times.push_back(milliseconds);
        };
        if (step % 2) { run(update, updateDirection, updateTimes); run(full, fullDirection, fullTimes); }
        else { run(full, fullDirection, fullTimes); run(update, updateDirection, updateTimes); }
        fullDirection.deviceCopyTo(difference); difference -= updateDirection;
        maxDirectionError = std::max(maxDirectionError, difference.normF() / fullDirection.normF());
        fullGamma.download(a); updateGamma.download(b);
        for (int j = 0; j < memory; ++j) maxCoefficientError = std::max(maxCoefficientError, std::abs(a[j] - b[j]));
        if (Session::getInstance().totalAllocatedBytes() != allocated) throw std::runtime_error("Benchmark allocated device memory");
    }
    if (profile) gpuErrChk(cudaProfilerStop());
    gpuErrChk(cudaEventDestroy(start)); gpuErrChk(cudaEventDestroy(stop));
    rapidjson::StringBuffer buffer; Writer out(buffer);
    out.StartObject(); out.Key("rows"); out.Int(rows); out.Key("memory"); out.Int(memory);
    out.Key("warmup_steps"); out.Int(warmup); out.Key("measured_steps"); out.Int(steps);
    out.Key("full_ms"); out.StartArray(); for (double value : fullTimes) out.Double(value); out.EndArray();
    out.Key("update_ms"); out.StartArray(); for (double value : updateTimes) out.Double(value); out.EndArray();
    out.Key("max_relative_direction_error"); out.Double(maxDirectionError);
    out.Key("max_absolute_coefficient_error"); out.Double(maxCoefficientError);
    out.Key("qr"); writeQrStatistics(out, update.qrStatistics());
    out.Key("device_allocation_unchanged"); out.Bool(true);
    out.EndObject(); std::cout << buffer.GetString() << '\n';
}

int main(int argc, char **argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--benchmark") {
            benchmark(std::stoi(argv[2]), std::stoi(argv[3])); return 0;
        }
        const bool single = argc == 2 && std::string(argv[1]) == "--float";
        std::string line;
        while (std::getline(std::cin, line)) {
            rapidjson::Document input; input.Parse(line.c_str());
            if (input.HasParseError() || !input.HasMember("frames") || input["frames"].Empty())
                throw std::invalid_argument("Invalid QR sequence");
            if (single) checkSequence<float>(input); else checkSequence<double>(input);
        }
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
