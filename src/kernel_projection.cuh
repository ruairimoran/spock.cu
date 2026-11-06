#ifndef SPOCK_KERNEL_PROJECTION_CUH
#define SPOCK_KERNEL_PROJECTION_CUH

#include "tree.cuh"

template<typename T>
__global__ void k_memCpyInTS(T *, T *, size_t, size_t, size_t, size_t *, size_t *);

template<typename T>
__global__ void k_memCpyOutTS(T *, T *, size_t, size_t, size_t, size_t *, size_t *);

namespace spock::detail {

// S2 acts only on y and the nonroot t/s entries. Input and output scratch
// have fixed roles, so borrowed views and captured device pointers stay valid.
template<typename T>
void projectRiskKernels(ScenarioTree<T> &tree, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                        DTensor<T> &input, DTensor<T> &output, DTensor<T> &projector) {
    const auto stream = commonStream(y, t, s, input, output, projector);
    memCpyNode2Node(input, y, 0, tree.numNonleafNodesMinus1(), y.numRows());
    k_memCpyInTS<<<numBlocks(tree.numNodes(), TPB), TPB, 0, stream>>>(
        input.raw(), t.raw(), tree.numNodes(), input.numRows(), y.numRows(),
        tree.d_ancestors().raw(), tree.d_childFrom().raw());
    k_memCpyInTS<<<numBlocks(tree.numNodes(), TPB), TPB, 0, stream>>>(
        input.raw(), s.raw(), tree.numNodes(), input.numRows(), y.numRows() + tree.numEvents(),
        tree.d_ancestors().raw(), tree.d_childFrom().raw());
    gpuErrChk(cudaGetLastError());
    output.addAB(projector, input);
    memCpyNode2Node(y, output, 0, tree.numNonleafNodesMinus1(), y.numRows());
    k_memCpyOutTS<<<numBlocks(tree.numNodes(), TPB), TPB, 0, stream>>>(
        t.raw(), output.raw(), tree.numNodes(), output.numRows(), y.numRows(),
        tree.d_ancestors().raw(), tree.d_childFrom().raw());
    k_memCpyOutTS<<<numBlocks(tree.numNodes(), TPB), TPB, 0, stream>>>(
        s.raw(), output.raw(), tree.numNodes(), output.numRows(), y.numRows() + tree.numEvents(),
        tree.d_ancestors().raw(), tree.d_childFrom().raw());
    gpuErrChk(cudaGetLastError());
}

class ProjectionEvent {
    cudaEvent_t m_event{};
public:
    ProjectionEvent() { gpuErrChk(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming)); }
    ~ProjectionEvent() { gpuErrChk(cudaEventDestroy(m_event)); }
    ProjectionEvent(const ProjectionEvent &) = delete;
    ProjectionEvent &operator=(const ProjectionEvent &) = delete;
    void record(cudaStream_t stream) { gpuErrChk(cudaEventRecord(m_event, stream)); }
    void waitOn(cudaStream_t stream) { gpuErrChk(cudaStreamWaitEvent(stream, m_event, 0)); }
};

// Owns events and non-owning S2 views; all numerical allocations belong to
// Cache/ProblemData. The main solver's views retain their original stream IDs.
template<typename T>
class ConcurrentPrimalProjections {
    DTensor<T> m_y, m_t, m_s, m_input, m_output, m_projector;
    cudaStream_t m_main{}, m_aux{};
    ProjectionEvent m_ready, m_done;

    static DTensor<T> view(DTensor<T> &tensor) {
        return DTensor<T>(tensor, 2, 0, tensor.numMats() - 1);
    }

public:
    ConcurrentPrimalProjections(cudaStream_t main, size_t streamIndex,
                               DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                               DTensor<T> &input, DTensor<T> &output, DTensor<T> &projector)
        : m_y(view(y)), m_t(view(t)), m_s(view(s)), m_input(view(input)),
          m_output(view(output)), m_projector(view(projector)), m_main(main) {
        for (auto *tensor : {&m_y, &m_t, &m_s, &m_input, &m_output, &m_projector})
            tensor->setStreamIdx(streamIndex);
        m_aux = commonStream(m_y, m_t, m_s, m_input, m_output, m_projector);
        if (m_aux == m_main)
            throw std::invalid_argument("Concurrent projections require distinct CUDA streams");
    }

    ~ConcurrentPrimalProjections() {
        // Cache destroys this object before releasing any backing allocations.
        // Drain both streams even if an exception interrupted submission.
        gpuErrChk(cudaStreamSynchronize(m_main));
        gpuErrChk(cudaStreamSynchronize(m_aux));
    }

    void enqueueRiskProjection(ScenarioTree<T> &tree) {
        m_ready.record(m_main);
        m_ready.waitOn(m_aux);
        projectRiskKernels(tree, m_y, m_t, m_s, m_input, m_output, m_projector);
        m_done.record(m_aux);
    }

    void join() { m_done.waitOn(m_main); }
};

}  // namespace spock::detail

#endif
