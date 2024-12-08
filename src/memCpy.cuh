#ifndef MEMCPY_CUH
#define MEMCPY_CUH

#include "../include/gpu.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyNode2Node(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyAnc2Node(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyLeaf2ZeroLeaf(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyZeroLeaf2Leaf(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyCh2Node(T *, T *, size_t, size_t, size_t, size_t, size_t *, size_t *, bool);

/**
 * Memory copy for trees
 */
enum MemCpyMode {
    node2Node,  ///< transfer node data to same node index
    anc2Node,  ///< transfer ancestor data to node index
    leaf2ZeroLeaf,  ///< transfer leaf data to zero-indexed leaf nodes
    zeroLeaf2Leaf,  ///< transfer zero-indexed leaf data to leaf nodes
    defaultMode = node2Node
};

TEMPLATE_WITH_TYPE_T
void memCpy(DTensor<T> *dst, DTensor<T> *src,
            size_t nodeFrom, size_t nodeTo, size_t numEl,
            size_t elFromDst = 0, size_t elFromSrc = 0,
            MemCpyMode mode = MemCpyMode::defaultMode,
            DTensor<size_t> *ancestors = nullptr) {
    size_t nodeSizeDst = dst->numRows();
    size_t nodeSizeSrc = src->numRows();
    if (dst->numCols() != 1 || src->numCols() != 1) throw std::invalid_argument("[memCpy] numCols must be 1.");
    if (std::max(nodeSizeDst, nodeSizeSrc) > TPB) throw std::invalid_argument("[memCpy] Node data too large.");
    if (mode == anc2Node && nodeFrom < 1) throw std::invalid_argument("[memCpy] Root node has no ancestor.");
    size_t nBlocks = nodeTo + 1;
    if (mode == node2Node) {
        k_memCpyNode2Node<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst, nodeSizeSrc,
                                            elFromDst, elFromSrc);
    }
    if (mode == anc2Node) {
        k_memCpyAnc2Node<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst, nodeSizeSrc,
                                           elFromDst, elFromSrc, ancestors->raw());
    }
    /**
     * For leaf transfers, you must transfer all leaf nodes! So `nodeFrom` == numNonleafNodes.
     * The `nodeFrom/To` requires the actual node numbers (not zero-indexed).
     */
    if (mode == leaf2ZeroLeaf) {
        k_memCpyLeaf2ZeroLeaf<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst,
                                                nodeSizeSrc, elFromDst, elFromSrc);
    }
    if (mode == zeroLeaf2Leaf) {
        k_memCpyZeroLeaf2Leaf<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst,
                                                nodeSizeSrc, elFromDst, elFromSrc);
    }
}


#endif
