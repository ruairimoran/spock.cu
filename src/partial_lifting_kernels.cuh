/**
 * @file partial_lifting_kernels.cuh
 * @brief CUDA kernels and device helpers for blocked dynamics projections.
 *
 * Let F_j be the closed-loop edge matrix A_j + B_j K_parent(j). The forward
 * recurrence applies F_j; the backward recurrence applies its transpose.
 * Every edge matrix is stored as F_j^T in column-major order, with leading
 * dimension `states`. The incoming edge of node j occupies matrix batch j.
 * Node-indexed vectors contain `states` consecutive values per global node.
 * All node indices are zero-based, with the tree root at node zero.
 *
 * The common unary suffix consists of `chains` paths, each containing `length`
 * edges and `length + 1` nodes. Its lookup table is chain-major:
 * @code
 * nodes[chain * (length + 1) + position]
 * @endcode
 * Segment summaries and block maps instead use level-major batch indexing:
 * @code
 * segment = level * chains + chain
 * @endcode
 * There are `ceil(length / blockLength)` segment levels; the final segment may
 * contain fewer than `blockLength` edges.
 *
 * @note All pointer arguments refer to device-accessible storage. Launches must
 * use the stream associated with the participating GPUtils tensors. The caller
 * must order dependent launches on that stream, or establish equivalent event
 * dependencies, and retain every buffer until its users have completed.
 */

#ifndef SPOCK_PARTIAL_LIFTING_KERNELS_CUH
#define SPOCK_PARTIAL_LIFTING_KERNELS_CUH

#include "../include/gpu.cuh"

namespace spock::detail {

/**
 * @brief Gather one incoming-edge factor from each chain into a dense batch.
 *
 * Copies F_j^T for the node at @p position on each chain without changing the
 * matrix layout. The packed batch feeds GPUtils' batched matrix products when
 * forming segment maps.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @param[out] packed Output batch of `chains` column-major matrices, each of
 *                    shape `states` by `states`.
 * @param[in] fTranspose Node-indexed batch of column-major F_j^T matrices.
 * @param[in] nodes Chain-major global-node lookup table; contains
 *                  `chains * (length + 1)` entries.
 * @param[in] states Number of state components, and matrix leading dimension.
 * @param[in] chains Number of independent unary chains.
 * @param[in] length Number of edges in each chain.
 * @param[in] position Edge's child position along each chain, in [1, length].
 *
 * @pre states, chains and length are positive; every selected node identifies
 * a valid matrix in @p fTranspose. Output and input matrix storage do not overlap.
 * @pre Launch a one-dimensional grid and block covering at least
 * `chains * states * states` threads. Extra threads are ignored.
 */
template<typename T>
__global__ void packChainFactors(T *packed, const T *fTranspose,
    const size_t *nodes, size_t states, size_t chains, size_t length, size_t position) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t matrixSize = states * states;
    if (index < chains * matrixSize) {
        const size_t chain = index / matrixSize;
        const size_t node = nodes[chain * (length + 1) + position];
        packed[index] = fTranspose[node * matrixSize + index % matrixSize];
    }
}

/**
 * @brief Form every nonroot forward offset B_j d_parent(j) + e_j.
 *
 * Each thread produces one component. These offsets are the additive sources
 * for the forward chain recurrence after the batched input solve. The root's
 * output vector is left untouched because its state is prescribed separately.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @param[out] offsets Node-indexed output vectors; storage for `nodes * states`
 *                     values is required, although the root is not written.
 * @param[in] ab Node-indexed batch of column-major [A_j B_j] matrices, each
 *               of shape `states` by `states + controls`.
 * @param[in] d Solved input-offset vectors, with `controls` consecutive values
 *              per nonleaf parent node.
 * @param[in] affine Optional node-indexed vectors e_j. Pass nullptr for linear
 *                   dynamics, which supplies a zero affine contribution.
 * @param[in] parents Global parent index for every node; entry zero is unused.
 * @param[in] states Number of state components per node.
 * @param[in] controls Number of input components per nonleaf node.
 * @param[in] nodes Total tree-node count, including the root.
 *
 * @pre states is positive, nodes is at least one, and every nonroot parent
 * index addresses a valid vector in @p d. The input solve has completed before
 * this kernel reads its output.
 * @pre Launch a one-dimensional grid and block covering at least
 * `(nodes - 1) * states` threads. Extra threads are ignored.
 */
template<typename T>
__global__ void dynamicsInputOffsets(T *offsets, const T *ab, const T *d,
    const T *affine, const size_t *parents, size_t states, size_t controls, size_t nodes) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= (nodes - 1) * states) return;
    const size_t node = 1 + index / states, row = index % states;
    T sum = affine ? affine[node * states + row] : T(0);
    for (size_t col = 0; col < controls; ++col)
        sum += ab[(node * (states + controls) + states + col) * states + row]
               * d[parents[node] * controls + col];
    offsets[node * states + row] = sum;
}

/**
 * @brief Reduce child contributions for one stage of the branching prefix.
 *
 * Computes y_i = sources_i + sum_j contributions_j over each parent's children.
 * The contributions F_j^T y_j are formed beforehand with a batched GPUtils
 * product, so this kernel traverses only the resulting child vectors.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @param[out] y Node-indexed destination vectors. Only parents in
 *               [first, first + count) are written.
 * @param[in] sources Node-indexed additive backward-source vectors.
 * @param[in] contributions Contiguous child-vector batch. Batch entry zero
 *                          corresponds to global node @p firstChild.
 * @param[in] childFrom Global index of the first child of each parent.
 * @param[in] children Number of contiguous children belonging to each parent.
 * @param[in] states Number of state components per vector.
 * @param[in] first Global index of the first parent to process.
 * @param[in] count Number of consecutive parent nodes to process.
 * @param[in] firstChild Global index represented by the first contribution.
 *
 * @pre Each requested parent's complete child range is represented in
 * @p contributions, and the preceding child products have completed.
 * @pre Launch a one-dimensional grid and block covering at least
 * `count * states` threads. Extra threads are ignored.
 */
template<typename T>
__global__ void partialPrefixBackward(T *y, const T *sources, const T *contributions,
    const size_t *childFrom, const size_t *children, size_t states,
    size_t first, size_t count, size_t firstChild) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * states) return;
    const size_t parent = first + index / states, row = index % states;
    T result = sources[parent * states + row];
    for (size_t ch = 0; ch < children[parent]; ++ch)
        result += contributions[(childFrom[parent] + ch - firstChild) * states + row];
    y[parent * states + row] = result;
}

/**
 * @brief Reduce one value from every lane of a complete 32-lane warp.
 *
 * @tparam T Real floating-point scalar type supported by CUDA shuffle
 *           intrinsics; float and double are supported.
 * @param[in] value This lane's contribution to the sum.
 * @return In lane zero, the sum of all 32 input contributions. Return values
 *         in the other lanes are not the complete warp sum and must not be
 *         used as such.
 *
 * @pre All 32 lanes are active and execute this call together. The fixed full
 * participation mask does not support a partially active warp.
 * @note The sum is not broadcast to the other lanes. No block-wide barrier is
 * performed by this helper.
 */
template<typename T>
__device__ T warpSum(T value) {
    for (int shift = 16; shift > 0; shift >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, shift);
    return value;
}

/**
 * @brief Propagate one forward stage of the branching prefix.
 *
 * Computes x_j = F_j x_parent(j) + offsets_j. One warp cooperates on each
 * output component, with contiguous reads from the stored column-major F_j^T.
 * Different CUDA blocks process independent child/component tiles.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @param[in,out] statesOut Node-indexed state vectors. Parent states are read;
 *                          child states in [first, first + count) are written.
 * @param[in] offsets Node-indexed additive forward-offset vectors.
 * @param[in] fTranspose Node-indexed batch of column-major F_j^T matrices,
 *                       each of shape `states` by `states`.
 * @param[in] parents Global parent index for every node.
 * @param[in] states Number of state components per node.
 * @param[in] first Global index of the first child to process.
 * @param[in] count Number of consecutive children to process.
 *
 * @pre All requested children belong to one stage and their parent states are
 * complete before launch; this kernel does not synchronize dependent stages.
 * @pre Use block dimensions `(32, W, 1)` for a positive warp count W, with
 * `grid.x >= count`, `grid.y = ceil(states / W)` and `grid.z = 1`.
 * Out-of-range child blocks and component tiles are ignored.
 */
template<typename T>
__global__ void partialPrefixForward(T *statesOut, const T *offsets, const T *fTranspose,
    const size_t *parents, size_t states, size_t first, size_t count) {
    const size_t child = first + blockIdx.x;
    const size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (blockIdx.x >= count || row >= states) return;
    T sum = 0;
    for (size_t col = threadIdx.x; col < states; col += 32)
        sum += fTranspose[(child * states + row) * states + col]
               * statesOut[parents[child] * states + col];
    sum = warpSum(sum);
    if (threadIdx.x == 0) statesOut[child * states + row] = sum + offsets[child * states + row];
}

/**
 * @brief Apply one affine edge recurrence within a local chain segment.
 *
 * The forward path computes result = F * value + source, using one warp per
 * output component. The backward path computes result = F^T * value + source,
 * assigning each output component to a thread with a serial column loop.
 * Both paths read the same column-major representation of F^T.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @tparam Forward If true, apply F; if false, apply F^T.
 * @param[in] fTranspose One column-major `states` by `states` matrix F^T.
 * @param[in] value Input state vector, containing `states` values. The local
 *                  recurrence supplies this vector from shared memory.
 * @param[out] result Destination vector, containing `states` values. It must
 *                    not overlap @p value or @p fTranspose; local recurrences
 *                    use a second shared-memory vector for this output.
 * @param[in] source Additive source vector, containing `states` values.
 * @param[in] states Number of vector components and matrix leading dimension.
 *
 * @pre Use a block layout `(32, W, 1)` with W positive. All participating
 * threads call this helper together, and every input component is ready.
 * @note This is a device helper, not a separately launched kernel. It provides
 * no block-wide synchronization. The caller must synchronize before consuming
 * another thread's output or reusing either shared vector as a destination.
 * @see localChainRecurrence
 */
template<typename T, bool Forward>
__device__ void chainMatVec(const T *fTranspose, const T *value, T *result,
    const T *source, size_t states) {
    if constexpr (Forward) {
        for (size_t row = threadIdx.y; row < states; row += blockDim.y) {
            T sum = 0;
            for (size_t col = threadIdx.x; col < states; col += 32)
                sum += fTranspose[row * states + col] * value[col];
            sum = warpSum(sum);
            if (threadIdx.x == 0) result[row] = sum + source[row];
        }
    } else {
        const size_t thread = threadIdx.y * 32 + threadIdx.x;
        for (size_t row = thread; row < states; row += size_t(32) * blockDim.y) {
            T sum = 0;
            for (size_t col = 0; col < states; ++col)
                sum += fTranspose[col * states + row] * value[col];
            result[row] = sum + source[row];
        }
    }
}

/**
 * @brief Compute segment summaries or reconstruct segment-interior vectors.
 *
 * Each CUDA block owns one short segment. Let left and right denote its
 * boundary nodes and let C be the product of forward edge matrices from left
 * to right. In summary mode, the kernel computes only the additive term h in
 * `x_right = C * x_left + h` (forward) or
 * `y_left = C^T * y_right + h` (backward), independently of boundary values.
 * The maps C^T are prepared separately and applied by chainBoundary().
 *
 * In reconstruction mode, the appropriate starting boundary is already known.
 * The kernel propagates from it and writes only strictly interior nodes. Both
 * endpoints remain untouched, preventing adjacent segments from racing on their
 * shared boundary. A one-edge segment has no interior nodes to reconstruct.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @tparam Forward If true, follow the chain from left to right using F_j;
 *                 otherwise follow it from right to left using F_j^T.
 * @tparam Reconstruct If false, produce an additive segment summary from
 *                     @p sources. If true, reconstruct interior values from
 *                     the known boundary in @p values.
 * @param[in,out] values Node-indexed vectors: forward states or backward y
 *                       values. In reconstruction mode, the starting boundary
 *                       is read and interior nodes are written. Unused in
 *                       summary mode and may then be nullptr.
 * @param[out] summaries Segment-major additive vectors, with `states` values
 *                       per `level * chains + chain`. Written only in summary
 *                       mode; may be nullptr in reconstruction mode.
 * @param[in] sources Node-indexed source vectors: forward offsets B_j d_parent
 *                    + e_j, or the additive sources of the backward recurrence.
 * @param[in] fTranspose Node-indexed batch of column-major F_j^T matrices.
 * @param[in] nodes Chain-major node lookup table containing
 *                  `chains * (length + 1)` global node indices.
 * @param[in] states Number of components per vector and matrix leading dimension.
 * @param[in] chains Number of independent unary chains.
 * @param[in] length Number of edges in each chain, excluding its starting node.
 * @param[in] blockLength Nominal positive edge count per segment. The final
 *                        segment uses the remaining edges when shorter.
 *
 * @pre states, chains, length and blockLength are positive; segment edge counts
 * fit in an int. Every lookup-table entry addresses valid node storage, and
 * each segment owns its interior nodes exclusively.
 * @pre Launch exactly `chains * ceil(length / blockLength)` CUDA blocks in
 * `grid.x`, with `grid.y = grid.z = 1`. There is no extra-block bounds check.
 * Use block dimensions `(32, W, 1)` with W positive and allocate at least
 * `2 * states * sizeof(T)` bytes of dynamic shared memory, within the device's
 * per-block limit. The host launcher uses W=8 forward and a configured W=4 or 8
 * backward, selected before graph capture.
 * @pre In reconstruction mode, the coarse boundary sweep has completed before
 * launch. Summary mode requires only the source and edge-factor arrays.
 * @note Synchronization inside a segment protects its two shared vectors;
 * dependencies between summary, boundary and reconstruction launches must be
 * established by the caller.
 * @see chainMatVec
 * @see chainBoundary
 */
template<typename T, bool Forward, bool Reconstruct>
__global__ void localChainRecurrence(T *values, T *summaries, const T *sources,
    const T *fTranspose, const size_t *nodes, size_t states, size_t chains,
    size_t length, size_t blockLength) {
    extern __shared__ __align__(16) unsigned char sharedStorage[];
    T *value = reinterpret_cast<T *>(sharedStorage), *next = value + states;
    const size_t thread = threadIdx.y * 32 + threadIdx.x;
    const size_t block = blockIdx.x, chain = block % chains, level = block / chains;
    const size_t start = level * blockLength;
    const int edges = int(min(blockLength, length - start));
    const size_t *path = nodes + chain * (length + 1) + start;
    const size_t initial = Forward ? path[Reconstruct ? 0 : 1]
                                   : path[Reconstruct ? edges : edges - 1];
    for (size_t row = thread; row < states; row += size_t(32) * blockDim.y)
        value[row] = Reconstruct ? values[initial * states + row] : sources[initial * states + row];
    __syncthreads();
    const int begin = Forward ? (Reconstruct ? 1 : 2) : (Reconstruct ? edges - 1 : edges - 2);
    const int end = Forward ? (Reconstruct ? edges - 1 : edges) : (Reconstruct ? 1 : 0);
    for (int pos = begin; Forward ? pos <= end : pos >= end; pos += Forward ? 1 : -1) {
        const size_t factorNode = path[Forward ? pos : pos + 1];
        chainMatVec<T, Forward>(fTranspose + factorNode * states * states,
            value, next, sources + path[pos] * states, states);
        __syncthreads();
        T *previous = value; value = next; next = previous;
        if constexpr (Reconstruct)
            for (size_t row = thread; row < states; row += size_t(32) * blockDim.y)
                values[path[pos] * states + row] = value[row];
        __syncthreads();
    }
    if constexpr (!Reconstruct)
        for (size_t row = thread; row < states; row += size_t(32) * blockDim.y)
            summaries[block * states + row] = value[row];
}

/**
 * @brief Advance one coarse segment level across all independent chains.
 *
 * For blockLength > 1, applies the prepared segment map and additive summary:
 * `x_right = C * x_left + h` forward, or
 * `y_left = C^T * y_right + h` backward. The map is stored as column-major C^T.
 * For blockLength == 1, reads the original edge factor and node source directly,
 * providing the reordered stage baseline without segment-summary storage.
 *
 * @tparam T Real floating-point scalar type; float and double are supported.
 * @tparam Forward If true, read the left boundary and write the right boundary;
 *                 otherwise read the right boundary and write the left boundary.
 * @param[in,out] values Node-indexed forward states or backward y vectors.
 *                       Only the destination boundary at this level is written.
 * @param[in] sources Node-indexed additive vectors used only when
 *                    blockLength == 1; may be nullptr otherwise.
 * @param[in] summaries Segment-major additive vectors used when blockLength > 1;
 *                      contains `states` values per segment. May be nullptr for
 *                      the blockLength == 1 baseline.
 * @param[in] maps Segment-major batch of column-major C^T matrices, each of
 *                 shape `states` by `states`. Required when blockLength > 1;
 *                 may be nullptr for the blockLength == 1 baseline.
 * @param[in] fTranspose Node-indexed F_j^T matrix batch, used only for the
 *                       blockLength == 1 baseline; may be nullptr otherwise.
 * @param[in] nodes Chain-major node lookup table containing
 *                  `chains * (length + 1)` global node indices.
 * @param[in] states Number of components per vector and matrix leading dimension.
 * @param[in] chains Number of independent unary chains.
 * @param[in] length Number of edges in each chain.
 * @param[in] blockLength Nominal positive edge count per segment.
 * @param[in] level Zero-based segment level, strictly less than
 *                  `ceil(length / blockLength)`.
 *
 * @pre states, chains, length and blockLength are positive.
 * @pre Input boundary values are ready; for blockLength > 1, segment summaries
 * and maps have been computed using the current sources and edge factors.
 * @pre Launch exactly `chains` blocks in grid.x and use block dimensions
 * `(32, 4, 1)`. The row-tile indexing requires this block layout. Set grid.y to
 * `ceil(states / 4)` forward or `ceil(states / 128)` backward, and grid.z to one.
 * @note Launch levels in increasing order forward and decreasing order backward,
 * using stream ordering or equivalent dependencies. Chains within a level are
 * independent; dependent levels cannot be computed in the same launch.
 * @see localChainRecurrence
 */
template<typename T, bool Forward>
__global__ void chainBoundary(T *values, const T *sources, const T *summaries,
    const T *maps, const T *fTranspose, const size_t *nodes,
    size_t states, size_t chains, size_t length, size_t blockLength, size_t level) {
    const size_t chain = blockIdx.x, start = level * blockLength;
    const size_t end = min(start + blockLength, length);
    const size_t *path = nodes + chain * (length + 1);
    const size_t block = level * chains + chain;
    const size_t left = path[start], right = path[end];
    const T *matrix = blockLength == 1 ? fTranspose + right * states * states
                                      : maps + block * states * states;
    const size_t sourceNode = Forward ? right : left;
    const T *offset = blockLength == 1 ? sources + sourceNode * states : summaries + block * states;
    if constexpr (Forward) {
        const size_t row = size_t(blockIdx.y) * 4 + threadIdx.y;
        if (row >= states) return;
        T sum = 0;
        for (size_t col = threadIdx.x; col < states; col += 32)
            sum += matrix[row * states + col] * values[left * states + col];
        sum = warpSum(sum);
        if (threadIdx.x == 0) values[right * states + row] = offset[row] + sum;
    } else {
        const size_t row = size_t(blockIdx.y) * 128 + threadIdx.y * 32 + threadIdx.x;
        if (row >= states) return;
        T sum = 0;
        for (size_t col = 0; col < states; ++col)
            sum += matrix[col * states + row] * values[right * states + col];
        values[left * states + row] = offset[row] + sum;
    }
}

}  // namespace spock::detail
#endif
