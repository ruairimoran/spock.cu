#ifndef SPOCK_CUDA_GRAPH_CUH
#define SPOCK_CUDA_GRAPH_CUH

#include "../include/gpu.cuh"
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace spock {

enum class GraphMode { none, dynamics, cp };

inline const char *graphModeName(GraphMode mode) {
    switch (mode) {
        case GraphMode::none: return "none";
        case GraphMode::dynamics: return "s1";
        case GraphMode::cp: return "cp";
    }
    return "unknown";
}

struct GraphStatistics {
    size_t nodes = 0;
    size_t allocationNodes = 0;
    size_t launches = 0;
    double setupSeconds = 0.;
};

namespace detail {

inline void checkGraphCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Owns a fixed-address graph and executable. Numerical inputs stay in their
// original device allocations; replay reads their current contents.
class CudaGraph {
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_executable{};
    cudaStream_t m_stream{};
    GraphStatistics m_statistics;

public:
    CudaGraph() = default;
    CudaGraph(const CudaGraph &) = delete;
    CudaGraph &operator=(const CudaGraph &) = delete;
    ~CudaGraph() {
        if (m_executable) {
            // Complete pending launches before Cache releases borrowed storage.
            gpuErrChk(cudaStreamSynchronize(m_stream));
            gpuErrChk(cudaGraphExecDestroy(m_executable));
        }
        if (m_graph) gpuErrChk(cudaGraphDestroy(m_graph));
    }

    template<typename Submit>
    void capture(cudaStream_t stream, Submit &&submit) {
        if (m_graph) throw std::logic_error("CUDA graph is already captured");
        cudaStreamCaptureStatus status;
        checkGraphCuda(cudaStreamIsCapturing(stream, &status), "Query stream capture");
        if (status != cudaStreamCaptureStatusNone)
            throw std::invalid_argument("Cannot initialise a graph inside another stream capture");
        checkGraphCuda(cudaStreamSynchronize(stream), "Complete graph setup work");
        m_stream = stream;
        checkGraphCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "Begin graph capture");
        try {
            submit();
        } catch (...) {
            // Even an invalidated capture must be ended before normal CUDA
            // operations resume on this stream.
            cudaGraph_t discarded{};
            cudaStreamEndCapture(stream, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
            throw;
        }
        checkGraphCuda(cudaStreamEndCapture(stream, &m_graph), "End graph capture");
        checkGraphCuda(cudaGraphInstantiate(&m_executable, m_graph, 0), "Instantiate CUDA graph");
        checkGraphCuda(cudaGraphGetNodes(m_graph, nullptr, &m_statistics.nodes), "Count graph nodes");
        std::vector<cudaGraphNode_t> nodes(m_statistics.nodes);
        checkGraphCuda(cudaGraphGetNodes(m_graph, nodes.data(), &m_statistics.nodes), "Read graph nodes");
        for (const auto node : nodes) {
            cudaGraphNodeType type;
            checkGraphCuda(cudaGraphNodeGetType(node, &type), "Read graph node type");
            if (type == cudaGraphNodeTypeMemAlloc) ++m_statistics.allocationNodes;
        }
        checkGraphCuda(cudaGraphUpload(m_executable, stream), "Upload CUDA graph");
        checkGraphCuda(cudaStreamSynchronize(stream), "Complete CUDA graph upload");
    }

    void launch() {
        checkGraphCuda(cudaGraphLaunch(m_executable, m_stream), "Launch CUDA graph");
        ++m_statistics.launches;
    }

    void resetLaunchCount() { m_statistics.launches = 0; }
    void setSetupSeconds(double seconds) { m_statistics.setupSeconds = seconds; }
    const GraphStatistics &statistics() const { return m_statistics; }
};

}  // namespace detail
}  // namespace spock

#endif
