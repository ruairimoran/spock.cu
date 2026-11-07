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
    if (settings.HasMember("fit_backend")) {
        if (!settings["fit_backend"].IsString()) throw std::invalid_argument("Fit backend must be a string");
        const std::string backend = settings["fit_backend"].GetString();
        if (backend != "gpu" && backend != "cpu") throw std::invalid_argument("Unknown Anderson fit backend");
        options.fitBackend = backend == "cpu" ? AndersonFitBackend::cpu : AndersonFitBackend::gpu;
    }
    if (!options.enabled && options.fitBackend == AndersonFitBackend::cpu)
        throw std::invalid_argument("CPU reduced fitting requires regularised Anderson");
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

// JSON lines followed, when update=true, by the six little-endian float64
// arrays advertised in ready.fields. P/APB are not part of the update payload.
int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: networked_mpc_persistent DATA_DIRECTORY TOLERANCE MAX_TIME_SECONDS [ANDERSON_JSON] [--projections sequential|concurrent] [--cuda-graphs none|s1|cp] [--partial-lifting 0|1|2|3|4|5|8]\n";
        return 2;
    }
    static_assert(std::endian::native == std::endian::little);
    try {
        std::ios::sync_with_stdio(false);
        std::string path = argv[1];
        if (path.back() != '/') path += '/';
        const double tol = std::stod(argv[2]), maxTime = std::stod(argv[3]);
        const char *andersonJson = nullptr;
        auto projections = spock::ProjectionMode::sequential;
        auto graphs = spock::GraphMode::none;
        size_t partialBlock = 0;
        for (int arg = 4; arg < argc; ++arg) {
            const std::string option = argv[arg];
            if (option == "--projections") {
                if (++arg >= argc) throw std::invalid_argument("Missing projection mode");
                const std::string mode = argv[arg];
                if (mode == "sequential") projections = spock::ProjectionMode::sequential;
                else if (mode == "concurrent") projections = spock::ProjectionMode::concurrent;
                else throw std::invalid_argument("Unknown projection mode");
            } else if (option == "--partial-lifting") {
                if (++arg >= argc) throw std::invalid_argument("Missing partial-lifting block size");
                const std::string value = argv[arg];
                if (value != "0" && value != "1" && value != "2" && value != "3"
                    && value != "4" && value != "5" && value != "8")
                    throw std::invalid_argument("Partial lifting accepts 0, 1, 2, 3, 4, 5 or 8");
                partialBlock = std::stoul(value);
            } else if (option == "--cuda-graphs") {
                if (++arg >= argc) throw std::invalid_argument("Missing CUDA graph mode");
                const std::string mode = argv[arg];
                if (mode == "none") graphs = spock::GraphMode::none;
                else if (mode == "s1") graphs = spock::GraphMode::dynamics;
                else if (mode == "cp") graphs = spock::GraphMode::cp;
                else throw std::invalid_argument("Unknown CUDA graph mode");
            } else if (!andersonJson && !option.starts_with("--")) andersonJson = argv[arg];
            else throw std::invalid_argument("Unexpected persistent MPC argument");
        }
        if (graphs != spock::GraphMode::none && projections != spock::ProjectionMode::sequential)
            throw std::invalid_argument("CUDA graphs currently require sequential projections");
        const AndersonOptions anderson = andersonJson ? readAndersonOptions(andersonJson) : AndersonOptions{};
        // Session is configured before ScenarioTree allocates its first tensor.
        Session::setStreams(projections == spock::ProjectionMode::concurrent ? 2 : 1);
        const auto setupStart = Clock::now();
        checked(cudaFree(nullptr));
        ScenarioTree<double> tree(path);
        ProblemData<double> problem(tree);
        problem.dynamics()->configurePartialLifting(partialBlock);
        const auto lifting = problem.dynamics()->partialLiftingStatistics();
        CacheBuilder<double> builder(tree, problem);
        builder.tol(tol).maxTimeSecs(maxTime).projectionMode(projections).cudaGraphs(graphs);
        if (anderson.enabled) builder.regularisedAnderson(anderson);
        else builder.legacyAnderson();
        auto cache = builder.build();
        const auto stream = spock::executionStream(cache.solution());
        auto tensors = problem.dynamics()->linearUpdateTensors();
        const auto &names = Dynamics<double>::linearUpdateNames;
        size_t payloadBytes = 0;
        for (const auto *tensor : tensors) payloadBytes += tensor->numEl()*sizeof(double);
        size_t largest = 0;
        for (auto *tensor : tensors) largest = std::max(largest, tensor->numEl());
        std::vector<double> staging(largest);
        std::vector<double> state(tree.numStates());
        const size_t allocated = Session::getInstance().totalAllocatedBytes();
        cudaEvent_t start, stop;
        checked(cudaEventCreate(&start));
        checked(cudaEventCreate(&stop));
        checked(cudaStreamSynchronize(stream));
        const double setupSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();
        rapidjson::StringBuffer readyBuffer;
        rapidjson::Writer<rapidjson::StringBuffer> ready(readyBuffer);
        ready.StartObject();
        ready.Key("ready"); ready.Bool(true);
        ready.Key("dynamics_update_protocol"); ready.String("linear-v2");
        ready.Key("dynamics_update_bytes"); ready.Uint64(payloadBytes);
        ready.Key("partial_lifting_block_size"); ready.Uint64(partialBlock);
        ready.Key("partial_lifting_active"); ready.Bool(lifting.chains != 0);
        ready.Key("partial_lifting_device_bytes"); ready.Uint64(lifting.allocatedBytes);
        ready.Key("partial_lifting_matrix_bytes"); ready.Uint64(lifting.matrixBytes);
        ready.Key("partial_lifting_scratch_bytes"); ready.Uint64(lifting.scratchBytes);
        ready.Key("partial_lifting_blocks"); ready.Uint64(lifting.blocks);
        ready.Key("partial_lifting_chain_length"); ready.Uint64(lifting.chainLength);
        ready.Key("partial_lifting_tail_stage"); ready.Uint64(lifting.tailStage);
        ready.Key("projection_mode"); ready.String(spock::projectionModeName(cache.projectionMode()));
        ready.Key("cuda_graphs"); ready.String(spock::graphModeName(cache.graphMode()));
        ready.Key("graph_nodes"); ready.Uint64(cache.graphStatistics().nodes);
        ready.Key("graph_allocation_nodes"); ready.Uint64(cache.graphStatistics().allocationNodes);
        ready.Key("graph_setup_wall_s"); ready.Double(cache.graphStatistics().setupSeconds);
        ready.Key("setup_wall_s"); ready.Double(setupSeconds);
        ready.Key("device_allocated_bytes"); ready.Uint64(allocated);
        ready.Key("primal_size"); ready.Uint64(cache.sizePrimal());
        ready.Key("dual_size"); ready.Uint64(cache.sizeDual());
        ready.Key("anderson_mode"); ready.String(anderson.enabled ? "regularized" : "legacy");
        ready.Key("aa_regularization"); ready.Double(cache.andersonOptions().regularization);
        ready.Key("aa_rank_tolerance"); ready.Double(cache.andersonOptions().rankTolerance);
        ready.Key("aa_fit_backend"); ready.String(andersonFitBackendName(cache.andersonOptions().fitBackend));
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
            double liftingUpdateSeconds = 0.;
            if (request["update"].GetBool()) {
                if (!request.HasMember("sizes") || !request["sizes"].IsArray() || request["sizes"].Size() != tensors.size())
                        throw std::runtime_error("Dynamics payload requires the six fields advertised by linear-v2");
                for (size_t i = 0; i < tensors.size(); ++i)
                    if (!request["sizes"][i].IsUint64() || request["sizes"][i].GetUint64() != tensors[i]->numEl())
                        throw std::runtime_error("Dynamics update dimensions changed");
                for (auto *tensor : tensors) {
                    const size_t bytes = tensor->numEl() * sizeof(double);
                    if (!std::cin.read(reinterpret_cast<char *>(staging.data()), bytes))
                        throw std::runtime_error("Incomplete binary dynamics payload");
                    spock::detail::copyToDevice(tensor->raw(), staging.data(), bytes, stream);
                }
            }
            checked(cudaStreamSynchronize(stream));
            if (request["update"].GetBool() && lifting.matrixBytes) {
                const auto refreshStart = Clock::now();
                problem.dynamics()->refreshProjectionFactors();
                checked(cudaStreamSynchronize(stream));
                liftingUpdateSeconds = std::chrono::duration<double>(Clock::now() - refreshStart).count();
            }
            const double updateSeconds = std::chrono::duration<double>(Clock::now() - updateStart).count();
            checked(cudaEventRecord(start, stream));
            const auto solveStart = Clock::now();
            const int status = cache.runSpockResident(state, warm);
            checked(cudaEventRecord(stop, stream));
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
            out.Key("projection_mode"); out.String(spock::projectionModeName(cache.projectionMode()));
            out.Key("cuda_graphs"); out.String(spock::graphModeName(cache.graphMode()));
            out.Key("graph_launches"); out.Uint64(cache.graphStatistics().launches);
            out.Key("graph_nodes"); out.Uint64(cache.graphStatistics().nodes);
            out.Key("graph_setup_wall_s"); out.Double(solves == 0 ? cache.graphStatistics().setupSeconds : 0.);
            out.Key("status"); out.Int(status);
            out.Key("warm_start"); out.Bool(warm);
            out.Key("iterations"); out.Uint64(cache.solveIter());
            out.Key("gpu_time_s"); out.Double(milliseconds / 1000.0);
            out.Key("solver_wall_s"); out.Double(solveSeconds);
            out.Key("setup_wall_s"); out.Double(solves == 0 ? setupSeconds : 0.0);
            out.Key("partial_lifting_block_size"); out.Uint64(partialBlock);
            out.Key("partial_lifting_update_wall_s"); out.Double(liftingUpdateSeconds);
            out.Key("update_wall_s"); out.Double(updateSeconds);
            out.Key("dynamics_update_bytes"); out.Uint64(request["update"].GetBool() ? payloadBytes : 0);
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
