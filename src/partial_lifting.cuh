#ifndef SPOCK_PARTIAL_LIFTING_CUH
#define SPOCK_PARTIAL_LIFTING_CUH

#include "tree.cuh"
#include "partial_lifting_kernels.cuh"
#include <algorithm>

namespace spock {

struct PartialLiftingStatistics {
    size_t blockSize = 0, tailStage = 0, chainLength = 0, chains = 0, blocks = 0;
    size_t matrixBytes = 0, scratchBytes = 0, allocatedBytes = 0;
};

// Experimental blocking of the common unary suffix. Topology and storage are
// fixed at construction; refresh() changes only matrix values, keeping captured
// graph addresses valid. The stage path is retained when the suffix is empty.
template<typename T>
class PartialLifting {
    ScenarioTree<T> &m_tree;
    PartialLiftingStatistics m_stats;
    size_t m_levels = 0;
    std::unique_ptr<DTensor<size_t>> m_nodes;
    std::unique_ptr<DTensor<T>> m_maps, m_factorScratch, m_productScratch, m_offsets, m_prefixScratch;

    template<bool Forward, bool Reconstruct>
    void local(DTensor<T> &values, const DTensor<T> &sources, const DTensor<T> &fTranspose) {
        detail::localChainRecurrence<T, Forward, Reconstruct><<<m_stats.blocks, dim3(32, Forward ? 8 : 4),
            2 * m_tree.numStates() * sizeof(T), executionStream(values)>>>(
                values.raw(), m_offsets->raw(), sources.raw(), fTranspose.raw(), m_nodes->raw(),
                m_tree.numStates(), m_stats.chains, m_stats.chainLength, m_stats.blockSize);
        gpuErrChk(cudaPeekAtLastError());
    }

    template<bool Forward>
    void boundary(DTensor<T> &values, const DTensor<T> &sources,
                  const DTensor<T> &fTranspose, size_t level) {
        const size_t rows = Forward ? 4 : 128;
        detail::chainBoundary<T, Forward><<<dim3(m_stats.chains, numBlocks(m_tree.numStates(), rows)),
            dim3(32,4), 0, executionStream(values)>>>(values.raw(), sources.raw(),
                m_offsets ? m_offsets->raw() : nullptr, m_maps ? m_maps->raw() : nullptr,
                fTranspose.raw(), m_nodes->raw(), m_tree.numStates(), m_stats.chains,
                m_stats.chainLength, m_stats.blockSize, level);
        gpuErrChk(cudaPeekAtLastError());
    }

public:
    PartialLifting(ScenarioTree<T> &tree, DTensor<T> &fTranspose, size_t blockSize) : m_tree(tree) {
        if (blockSize != 1 && blockSize != 2 && blockSize != 3
            && blockSize != 4 && blockSize != 5 && blockSize != 8)
            throw std::invalid_argument("Partial lifting supports B=2/3/4/5/8 or the B=1 stage baseline");
        m_stats.blockSize = blockSize;
        for (size_t stage = 0; stage < tree.numStagesMinus1(); ++stage)
            for (size_t node = tree.stageFrom()[stage]; node <= tree.stageTo()[stage]; ++node)
                if (tree.numChildren()[node] != 1) m_stats.tailStage = stage + 1;
        m_stats.chainLength = tree.numStagesMinus1() - m_stats.tailStage;
        if (!m_stats.chainLength) return;
        int device, sharedLimit;
        gpuErrChk(cudaGetDevice(&device));
        gpuErrChk(cudaDeviceGetAttribute(&sharedLimit, cudaDevAttrMaxSharedMemoryPerBlock, device));
        if (blockSize > 1 && 2 * tree.numStates() * sizeof(T) > size_t(sharedLimit))
            throw std::invalid_argument("Partial lifting's two shared state vectors exceed this GPU's block limit");
        const size_t before = Session::getInstance().totalAllocatedBytes();
        m_stats.chains = tree.stageTo()[m_stats.tailStage] - tree.stageFrom()[m_stats.tailStage] + 1;
        m_levels = (m_stats.chainLength + blockSize - 1) / blockSize;
        m_stats.blocks = m_stats.chains * m_levels;
        std::vector<size_t> nodes;
        for (size_t root = tree.stageFrom()[m_stats.tailStage]; root <= tree.stageTo()[m_stats.tailStage]; ++root) {
            size_t node = root;
            nodes.push_back(node);
            for (size_t pos = 0; pos < m_stats.chainLength; ++pos) {
                node = tree.childFrom()[node];
                nodes.push_back(node);
            }
        }
        m_nodes = std::make_unique<DTensor<size_t>>(nodes, nodes.size());
        size_t prefixChildren = 0;
        for (size_t stage = 1; stage < m_stats.tailStage; ++stage)
            prefixChildren = std::max(prefixChildren,
                tree.stageTo()[stage + 1] - tree.stageFrom()[stage + 1] + 1);
        if (prefixChildren) {
            m_prefixScratch = std::make_unique<DTensor<T>>(tree.numStates(), 1, prefixChildren);
            m_stats.scratchBytes += m_prefixScratch->numEl() * sizeof(T);
        }
        if (blockSize > 1) {
            const size_t n = tree.numStates();
            m_maps = std::make_unique<DTensor<T>>(n, n, m_stats.blocks);
            // Reuse two chain-sized scratch batches at every coarse level,
            // rather than retaining a scratch matrix for every block.
            m_factorScratch = std::make_unique<DTensor<T>>(n, n, m_stats.chains);
            m_productScratch = std::make_unique<DTensor<T>>(n, n, m_stats.chains);
            m_offsets = std::make_unique<DTensor<T>>(n, 1, m_stats.blocks);
            m_stats.matrixBytes = m_maps->numEl() * sizeof(T);
            m_stats.scratchBytes += (m_factorScratch->numEl() + m_productScratch->numEl()) * sizeof(T);
            refresh(fTranspose);
        }
        m_stats.allocatedBytes = Session::getInstance().totalAllocatedBytes() - before;
    }

    bool active() const { return m_stats.chainLength != 0; }
    const PartialLiftingStatistics &statistics() const { return m_stats; }

    void refresh(DTensor<T> &fTranspose) {
        if (!m_maps) return;
        const auto stream = commonStream(fTranspose, *m_maps, *m_factorScratch, *m_productScratch, *m_nodes);
        const size_t n = m_tree.numStates(), count = m_stats.chains * n * n;
        auto pack = [&](size_t position) {
            detail::packChainFactors<<<numBlocks(count, size_t(256)), 256, 0, stream>>>(
                m_factorScratch->raw(), fTranspose.raw(), m_nodes->raw(), n,
                m_stats.chains, m_stats.chainLength, position);
            gpuErrChk(cudaPeekAtLastError());
        };
        for (size_t level = 0; level < m_levels; ++level) {
            DTensor<T> maps(*m_maps, 2, level * m_stats.chains, (level + 1) * m_stats.chains - 1);
            const size_t start = level * m_stats.blockSize;
            const size_t edges = std::min(m_stats.blockSize, m_stats.chainLength - start);
            pack(start + 1);
            m_factorScratch->deviceCopyToAsync(maps);
            for (size_t edge = 2; edge <= edges; ++edge) {
                pack(start + edge);
                // Stored C' = F_1' F_2' ... F_B'. GEMM output never aliases an input.
                m_productScratch->addAB(maps, *m_factorScratch);
                m_productScratch->deviceCopyToAsync(maps);
            }
        }
    }

    void backward(DTensor<T> &y, DTensor<T> &sources, DTensor<T> &fTranspose) {
        const auto stream = commonStream(y, sources, fTranspose, *m_nodes);
        sources.deviceCopyToAsync(y);  // Includes the terminal source at every leaf.
        if (m_stats.blockSize > 1) local<false, false>(y, sources, fTranspose);
        for (size_t level = m_levels; level-- > 0;) boundary<false>(y, sources, fTranspose, level);
        if (m_stats.blockSize > 1) local<false, true>(y, sources, fTranspose);
        for (size_t stage = m_stats.tailStage; stage-- > 0;) {
            if (stage == 0) continue;  // y at the fixed root is unused.
            const size_t first = m_tree.stageFrom()[stage], count = m_tree.stageTo()[stage] - first + 1;
            const size_t childFirst = m_tree.stageFrom()[stage + 1], childLast = m_tree.stageTo()[stage + 1];
            DTensor<T> factors(fTranspose, 2, childFirst, childLast);
            DTensor<T> children(y, 2, childFirst, childLast);
            DTensor<T> products(*m_prefixScratch, 2, 0, childLast - childFirst);
            products.addAB(factors, children);
            detail::partialPrefixBackward<<<numBlocks(count * m_tree.numStates(), size_t(256)), 256, 0, stream>>>(
                y.raw(), sources.raw(), products.raw(), m_tree.d_childFrom().raw(),
                m_tree.d_numChildren().raw(), m_tree.numStates(), first, count, childFirst);
            gpuErrChk(cudaPeekAtLastError());
        }
    }

    void forward(DTensor<T> &states, DTensor<T> &offsets, DTensor<T> &fTranspose) {
        const auto stream = commonStream(states, offsets, fTranspose, *m_nodes);
        for (size_t stage = 0; stage < m_stats.tailStage; ++stage) {
            const size_t first = m_tree.stageFrom()[stage + 1], count = m_tree.stageTo()[stage + 1] - first + 1;
            detail::partialPrefixForward<<<dim3(count, numBlocks(m_tree.numStates(), size_t(4))),
                dim3(32,4), 0, stream>>>(states.raw(), offsets.raw(), fTranspose.raw(),
                    m_tree.d_ancestors().raw(), m_tree.numStates(), first, count);
            gpuErrChk(cudaPeekAtLastError());
        }
        if (m_stats.blockSize > 1) local<true, false>(states, offsets, fTranspose);
        for (size_t level = 0; level < m_levels; ++level) boundary<true>(states, offsets, fTranspose, level);
        if (m_stats.blockSize > 1) local<true, true>(states, offsets, fTranspose);
    }
};

}  // namespace spock
#endif
