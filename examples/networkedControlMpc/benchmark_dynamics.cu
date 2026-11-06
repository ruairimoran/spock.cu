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
    if (argc != 6) {
        std::cerr << "Usage: benchmark_dynamics DATA OUTPUT s1|cp eager|graph SAMPLES\n";
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
    std::cout << "],\"device_bytes\":" << allocated << "}\n";
    cache.output(cp).saveToFile(argv[2]);
    gpuErrChk(cudaEventDestroy(begin)); gpuErrChk(cudaEventDestroy(end));
}
