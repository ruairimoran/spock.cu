#include <spock.cuh>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

using Clock = std::chrono::steady_clock;

static void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

static AndersonOptions readAndersonOptions(const char *json) {
    AndersonOptions options;
    rapidjson::Document settings;
    settings.Parse(json);
    if (settings.HasParseError() || !settings.IsObject() || !settings.HasMember("mode") || !settings["mode"].IsString())
        throw std::invalid_argument("Invalid Anderson settings JSON");
    const std::string mode = settings["mode"].GetString();
    if (mode != "legacy" && mode != "regularized") throw std::invalid_argument("Unknown Anderson mode");
    options.enabled = mode == "regularized";
    options.qr.incremental = options.enabled;
    if (settings.HasMember("qr_mode")) {
        if (!settings["qr_mode"].IsString()) throw std::invalid_argument("QR mode must be a string");
        const std::string qrMode = settings["qr_mode"].GetString();
        if (qrMode != "full" && qrMode != "update") throw std::invalid_argument("Unknown QR mode");
        options.qr.incremental = qrMode == "update";
    }
    if (!options.enabled && options.qr.incremental)
        throw std::invalid_argument("QR updates require regularised Anderson acceleration");
    for (auto entry : {std::pair{"qr_rebuild_interval", &options.qr.rebuildInterval},
                       std::pair{"qr_check_interval", &options.qr.checkInterval}}) {
        if (settings.HasMember(entry.first)) {
            if (!settings[entry.first].IsUint64()) throw std::invalid_argument("QR intervals must be positive integers");
            *entry.second = settings[entry.first].GetUint64();
        }
    }
    for (auto entry : {std::pair{"regularization", &options.regularization},
                       std::pair{"rank_tolerance", &options.rankTolerance},
                       std::pair{"max_coefficient_norm", &options.maxCoefficientNorm},
                       std::pair{"max_direction_ratio", &options.maxDirectionRatio},
                       std::pair{"qr_orthogonality_tolerance", &options.qr.orthogonalityTolerance}}) {
        if (settings.HasMember(entry.first)) {
            if (!settings[entry.first].IsNumber()) throw std::invalid_argument("Anderson settings must be numeric");
            *entry.second = settings[entry.first].GetDouble();
        }
    }
    if (options.enabled) options.validate();
    return options;
}

static void writeFinite(JsonWriter &out, double value) {
    if (std::isfinite(value)) out.Double(value);
    else out.Null();
}

static void writeAndersonDiagnostics(JsonWriter &out, const Cache<double> &cache) {
    const auto &stats = cache.andersonStatistics();
    out.Key("aa_fits"); out.Uint64(stats.fits);
    out.Key("aa_accepted"); out.Uint64(stats.accepted);
    out.Key("aa_filtered_columns"); out.Uint64(stats.filteredColumns);
    out.Key("aa_restarts"); out.Uint64(stats.restarts);
    out.Key("aa_bootstrap_steps"); out.Uint64(stats.bootstrapSteps);
    out.Key("aa_coefficient_rejections"); out.Uint64(stats.coefficientRejections);
    out.Key("aa_direction_rejections"); out.Uint64(stats.directionRejections);
    out.Key("aa_numerical_rejections"); out.Uint64(stats.numericalRejections);
    out.Key("aa_last_rank"); out.Uint64(stats.lastRank);
    out.Key("aa_last_scale"); writeFinite(out, stats.lastScale);
    out.Key("aa_last_coefficient_norm"); writeFinite(out, stats.lastCoefficientNorm);
    out.Key("aa_qr_rebuilds"); out.Uint64(stats.qr.rebuilds);
    out.Key("aa_qr_updates"); out.Uint64(stats.qr.updates);
    out.Key("aa_qr_reuses"); out.Uint64(stats.qr.reuses);
    out.Key("aa_qr_update_fallbacks"); out.Uint64(stats.qr.updateFallbacks);
    out.Key("aa_qr_orthogonality_checks"); out.Uint64(stats.qr.orthogonalityChecks);
    out.Key("aa_qr_last_orthogonality_error"); writeFinite(out, stats.qr.lastOrthogonalityError);
    out.Key("initial_residual"); writeFinite(out, cache.initialResidual());
    out.Key("final_residual"); writeFinite(out, cache.finalResidual());
    out.Key("effective_tolerance"); writeFinite(out, cache.effectiveTolerance());
    out.Key("cp_operator_evaluations"); out.Uint64(cache.operatorEvaluations());
    out.Key("backtracks"); out.Uint64(cache.backtracks());
}

// JSON lines followed, when update=true, by eight little-endian float64
// arrays in CUDA column-major order. No model files are read after setup.
int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: networked_mpc_persistent DATA_DIRECTORY TOLERANCE MAX_TIME_SECONDS [ANDERSON_JSON]\n";
        return 2;
    }
    static_assert(std::endian::native == std::endian::little);
    try {
        std::ios::sync_with_stdio(false);
        std::string path = argv[1];
        if (path.back() != '/') path += '/';
        const double tol = std::stod(argv[2]), maxTime = std::stod(argv[3]);
        const AndersonOptions anderson = argc == 5 ? readAndersonOptions(argv[4]) : AndersonOptions{};
        const auto setupStart = Clock::now();
        checked(cudaFree(nullptr));
        ScenarioTree<double> tree(path);
        ProblemData<double> problem(tree);
        CacheBuilder<double> builder(tree, problem);
        builder.tol(tol).maxTimeSecs(maxTime);
        if (anderson.enabled) builder.regularisedAnderson(anderson);
        else builder.legacyAnderson();
        auto cache = builder.build();
        auto tensors = problem.dynamics()->projectionTensors();
        const std::array<const char *, 8> names = {"BTr", "AB", "K", "KTr", "ABKTr", "P", "APB", "lowCholesky"};
        size_t largest = 0;
        for (auto *tensor : tensors) largest = std::max(largest, tensor->numEl());
        std::vector<double> staging(largest);
        std::vector<double> state(tree.numStates());
        const size_t allocated = Session::getInstance().totalAllocatedBytes();
        cudaEvent_t start, stop;
        checked(cudaEventCreate(&start));
        checked(cudaEventCreate(&stop));
        checked(cudaDeviceSynchronize());
        const double setupSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();
        rapidjson::StringBuffer readyBuffer;
        rapidjson::Writer<rapidjson::StringBuffer> ready(readyBuffer);
        ready.StartObject();
        ready.Key("ready"); ready.Bool(true);
        ready.Key("setup_wall_s"); ready.Double(setupSeconds);
        ready.Key("device_allocated_bytes"); ready.Uint64(allocated);
        ready.Key("primal_size"); ready.Uint64(cache.sizePrimal());
        ready.Key("dual_size"); ready.Uint64(cache.sizeDual());
        ready.Key("anderson_mode"); ready.String(anderson.enabled ? "regularized" : "legacy");
        ready.Key("aa_regularization"); ready.Double(cache.andersonOptions().regularization);
        ready.Key("aa_rank_tolerance"); ready.Double(cache.andersonOptions().rankTolerance);
        ready.Key("aa_qr_mode"); ready.String(cache.andersonOptions().qr.incremental ? "update" : "full");
        ready.Key("fields"); ready.StartArray();
        for (size_t i = 0; i < tensors.size(); ++i) {
            ready.StartObject();
            ready.Key("name"); ready.String(names[i]);
            ready.Key("shape"); ready.StartArray();
            ready.Uint64(tensors[i]->numRows()); ready.Uint64(tensors[i]->numCols()); ready.Uint64(tensors[i]->numMats());
            ready.EndArray(); ready.EndObject();
        }
        ready.EndArray(); ready.EndObject();
        std::cout << readyBuffer.GetString() << std::endl;

        size_t solves = 0;
        const bool profile = std::getenv("SPOCK_PROFILE_RESIDENT") != nullptr;
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto requestStart = Clock::now();
            rapidjson::Document request;
            request.Parse(line.c_str());
            if (request.HasParseError() || !request.HasMember("state") || !request["state"].IsArray()
                || request["state"].Size() != state.size() || !request.HasMember("warm_start")
                || !request["warm_start"].IsBool() || !request.HasMember("update") || !request["update"].IsBool())
                throw std::runtime_error("Invalid persistent MPC request");
            const bool warm = request["warm_start"].GetBool();
            if (warm && solves == 0) throw std::runtime_error("First solve must be cold");
            for (size_t i = 0; i < state.size(); ++i) {
                state[i] = request["state"][i].GetDouble();
                if (!std::isfinite(state[i])) throw std::runtime_error("Nonfinite initial state");
            }
            if (profile && solves == 1) checked(cudaProfilerStart());
            const auto updateStart = Clock::now();
            if (request["update"].GetBool()) {
                if (!request.HasMember("sizes") || !request["sizes"].IsArray() || request["sizes"].Size() != tensors.size())
                    throw std::runtime_error("Missing dynamics payload sizes");
                for (size_t i = 0; i < tensors.size(); ++i)
                    if (!request["sizes"][i].IsUint64() || request["sizes"][i].GetUint64() != tensors[i]->numEl())
                        throw std::runtime_error("Dynamics update dimensions changed");
                for (auto *tensor : tensors) {
                    const size_t bytes = tensor->numEl() * sizeof(double);
                    if (!std::cin.read(reinterpret_cast<char *>(staging.data()), bytes))
                        throw std::runtime_error("Incomplete binary dynamics payload");
                    checked(cudaMemcpy(tensor->raw(), staging.data(), bytes, cudaMemcpyHostToDevice));
                }
            }
            checked(cudaDeviceSynchronize());
            const double updateSeconds = std::chrono::duration<double>(Clock::now() - updateStart).count();
            checked(cudaEventRecord(start));
            const auto solveStart = Clock::now();
            const int status = cache.runSpockResident(state, warm);
            checked(cudaDeviceSynchronize());
            checked(cudaEventRecord(stop));
            checked(cudaEventSynchronize(stop));
            float milliseconds;
            checked(cudaEventElapsedTime(&milliseconds, start, stop));
            const double solveSeconds = std::chrono::duration<double>(Clock::now() - solveStart).count();
            std::vector<double> control;
            if (status == converged) control = cache.rootInput();
            for (double u : control) if (!std::isfinite(u)) throw std::runtime_error("Nonfinite root control");
            if (Session::getInstance().totalAllocatedBytes() != allocated)
                throw std::runtime_error("Persistent solver's net tensor allocation changed");
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> out(buffer);
            out.StartObject();
            out.Key("converged"); out.Bool(status == converged);
            out.Key("status"); out.Int(status);
            out.Key("warm_start"); out.Bool(warm);
            out.Key("iterations"); out.Uint64(cache.solveIter());
            out.Key("gpu_time_s"); out.Double(milliseconds / 1000.0);
            out.Key("solver_wall_s"); out.Double(solveSeconds);
            out.Key("setup_wall_s"); out.Double(solves == 0 ? setupSeconds : 0.0);
            out.Key("update_wall_s"); out.Double(updateSeconds);
            out.Key("worker_wall_s"); out.Double(std::chrono::duration<double>(Clock::now() - requestStart).count());
            out.Key("device_allocated_bytes"); out.Uint64(allocated);
            out.Key("solution_device_pointer"); out.Uint64(reinterpret_cast<uintptr_t>(cache.solution().raw()));
            out.Key("primal_size"); out.Uint64(cache.sizePrimal());
            out.Key("dual_size"); out.Uint64(cache.sizeDual());
            writeAndersonDiagnostics(out, cache);
            out.Key("control"); out.StartArray();
            for (double u : control) out.Double(u);
            out.EndArray(); out.EndObject();
            if (profile && solves == 1) checked(cudaProfilerStop());
            std::cout << buffer.GetString() << std::endl;
            ++solves;
        }
        checked(cudaEventDestroy(start));
        checked(cudaEventDestroy(stop));
    } catch (const std::exception &e) {
        std::cerr << "Persistent MPC worker failed: " << e.what() << std::endl;
        return 1;
    }
}
