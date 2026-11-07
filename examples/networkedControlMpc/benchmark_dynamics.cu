// Fixed inputs and existing workspaces; preparation is outside CUDA events.
#include "cache.cuh"
#include <chrono>
#include <cmath>
#include <iomanip>

class DynamicsBenchmarkCache : public Cache<double> {
public:
    DynamicsBenchmarkCache(ScenarioTree<double> &tree, ProblemData<double> &data)
        : Cache<double>(tree, data, 1e-3, 0., 0., 0, 8, 3, false, {},
                        spock::ProjectionMode::sequential) {}
    using Cache<double>::initialiseState;
    DTensor<double> &primalWorkspace() { return *m_d_workIteratePrim; }
    DTensor<double> &output(bool cp) { return cp ? *m_d_iterateCandidate : *m_d_workIterate; }
    void evaluate(bool cp) {
        if (cp) iter();
        else projectPrimalWorkspaceOnDynamics();
    }
};

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "Usage: benchmark_dynamics DATA OUTPUT s1|cp eager|graph SAMPLES [BLOCK_SIZE]\n";
        return 2;
    }
    std::string path = argv[1];
    if (path.back() != '/') path += '/';
    const std::string operation = argv[3], mode = argv[4];
    if ((operation != "s1" && operation != "cp") || (mode != "eager" && mode != "graph")) return 2;
    const int samples = std::stoi(argv[5]);
    if (samples < 1) return 2;
    const bool cp = operation == "cp";
    Session::setStreams(2);
    ScenarioTree<double> tree(path);
    ProblemData<double> data(tree);
    const size_t blockSize = argc == 7 ? std::stoul(argv[6]) : 0;
    const auto setupStream = spock::executionStream(tree.d_ancestors());
    gpuErrChk(cudaStreamSynchronize(setupStream));
    size_t freeBefore, totalMemory, freeAfter;
    gpuErrChk(cudaMemGetInfo(&freeBefore, &totalMemory));
    const auto setupStart = std::chrono::steady_clock::now();
    data.dynamics()->configurePartialLifting(blockSize);
    gpuErrChk(cudaStreamSynchronize(setupStream));
    gpuErrChk(cudaMemGetInfo(&freeAfter, &totalMemory));
    const double summarySetupMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setupStart).count();
    const auto lifting = data.dynamics()->partialLiftingStatistics();
    DynamicsBenchmarkCache cache(tree, data);
    std::vector<double> root(tree.numStates(), .1), values(cache.solution().numEl());
    for (size_t i = 0; i < values.size(); ++i) values[i] = .2 * std::sin(.17 * double(i));
    cache.initialiseState(root);
    cache.solution().upload(values);
    DTensor<double> originalPrimal(cache.solution(), 0, 0, cache.sizePrimal() - 1);
    const auto stream = spock::executionStream(cache.solution());
    auto prepare = [&] { if (!cp) originalPrimal.deviceCopyToAsync(cache.primalWorkspace()); };
    prepare(); cache.evaluate(cp);
    spock::detail::CudaGraph graph;
    if (mode == "graph") graph.capture(stream, [&] { cache.evaluate(cp); });
    auto evaluate = [&] {
        if (mode == "graph") graph.launch();
        else cache.evaluate(cp);
    };
    for (int i = 0; i < 10; ++i) { prepare(); evaluate(); }
    gpuErrChk(cudaStreamSynchronize(stream));
    const size_t allocated = Session::getInstance().totalAllocatedBytes();
    cudaEvent_t begin, end;
    gpuErrChk(cudaEventCreate(&begin)); gpuErrChk(cudaEventCreate(&end));
    std::cout << std::setprecision(17) << "{\"operation\":\"" << operation
              << "\",\"mode\":\"" << mode << "\",\"graph_nodes\":" << graph.statistics().nodes
              << ",\"samples\":[";
    for (int sample = 0; sample < samples; ++sample) {
        prepare();
        const auto wall = std::chrono::steady_clock::now();
        gpuErrChk(cudaEventRecord(begin, stream));
        evaluate();
        gpuErrChk(cudaEventRecord(end, stream));
        gpuErrChk(cudaEventSynchronize(end));
        float milliseconds;
        gpuErrChk(cudaEventElapsedTime(&milliseconds, begin, end));
        const double wallMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall).count();
        if (sample) std::cout << ',';
        std::cout << "{\"gpu_ms\":" << milliseconds << ",\"wall_ms\":" << wallMs << '}';
    }
    if (Session::getInstance().totalAllocatedBytes() != allocated)
        throw std::runtime_error("Projection benchmark changed tracked device allocation");
    const auto updateStart = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) data.dynamics()->refreshProjectionFactors();
    gpuErrChk(cudaStreamSynchronize(stream));
    const double refreshMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - updateStart).count() / 5;
    std::cout << "],\"device_bytes\":" << allocated
              << ",\"block_size\":" << blockSize << ",\"lifting_device_bytes\":" << lifting.allocatedBytes
              << ",\"lifting_cuda_setup_delta_bytes\":" << (freeBefore >= freeAfter ? freeBefore - freeAfter : 0)
              << ",\"lifting_matrix_bytes\":" << lifting.matrixBytes
              << ",\"lifting_scratch_bytes\":" << lifting.scratchBytes
              << ",\"lifting_blocks\":" << lifting.blocks
              << ",\"lifting_tail_stage\":" << lifting.tailStage
              << ",\"summary_setup_ms\":" << summarySetupMs
              << ",\"summary_refresh_ms\":" << refreshMs << "}\n";
    cache.output(cp).saveToFile(argv[2]);
    gpuErrChk(cudaEventDestroy(begin)); gpuErrChk(cudaEventDestroy(end));
}
