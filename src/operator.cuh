#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include <chrono>


TEMPLATE_WITH_TYPE_T
__global__ void k_setToZero(T *, size_t);

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
 * Memory copy mode for trees
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
            DTensor<size_t> *ancestors = nullptr,
            DTensor<size_t> *chFrom = nullptr, DTensor<size_t> *chTo = nullptr) {
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

static void constraintNotSupported() {
    throw std::invalid_argument("Constraint not supported.");
}


/**
 * Linear operator 'L' and its adjoint
 */
TEMPLATE_WITH_TYPE_T
class LinearOperator {
protected:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
    size_t m_matAxis = 2;
    size_t m_numNonleafNodesMinusOne = 0;
    size_t m_numNodesMinusOne = 0;
    size_t m_numLeafNodesMinusOne = 0;
    std::unique_ptr<DTensor<T>> m_d_uNonleafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xNonleafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uNonleafAdd = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xNonleafAdd = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xLeafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_scalarWorkspace = nullptr;

public:
    /**
     * Constructor
     */
    LinearOperator(ScenarioTree<T> &tree, ProblemData<T> &data) :
        m_tree(tree), m_data(data) {
        m_numNonleafNodesMinusOne = m_tree.numNonleafNodes() - 1;
        m_numNodesMinusOne = m_tree.numNodes() - 1;
        m_numLeafNodesMinusOne = m_tree.numLeafNodes() - 1;
        m_d_uNonleafWorkspace = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        m_d_xNonleafWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        m_d_uNonleafAdd = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNonleafNodes(), true);
        m_d_xNonleafAdd = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNonleafNodes(), true);
        m_d_xLeafWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numLeafNodes(), true);
        m_d_scalarWorkspace = std::make_unique<DTensor<T>>(1, 1, m_tree.numNodes(), true);
    }

    ~LinearOperator() {}

    /**
     * Public methods
     */
    void op(DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &,
            DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &);

    void adj(DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &,
             DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &);
};

template<typename T>
void LinearOperator<T>::op(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                           DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                           DTensor<T> &v, DTensor<T> &vi) {
    /* I */
    y.deviceCopyTo(i);
    /* II */
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_numNonleafNodesMinusOne);
    sNonleaf.deviceCopyTo(ii);
    ii.addAB(m_data.bTr(), y, -1., 1.);
    /* III */
    if (m_data.nonleafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
        memCpy(&iii, &x, 0, m_numNonleafNodesMinusOne, m_data.numStates());
        memCpy(&iii, &u, 0, m_numNonleafNodesMinusOne, m_data.numInputs(), m_data.numStates());
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply xuNonleaf by Gamma_{xu} */
    } else { constraintNotSupported(); }
    /* IV:1 */
    memCpy(m_d_xNonleafWorkspace.get(), &x, 1, m_numNodesMinusOne, m_data.numStates(), 0, 0,
           anc2Node, &m_tree.d_ancestors());
    m_d_xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *m_d_xNonleafWorkspace);
    /* IV:2 */
    memCpy(m_d_uNonleafWorkspace.get(), &u, 1, m_numNodesMinusOne, m_data.numInputs(), 0, 0,
           anc2Node, &m_tree.d_ancestors());
    m_d_uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *m_d_uNonleafWorkspace);
    /* IV:3,4 */
    t *= 0.5;  // This affects the current 't'!!! But it shouldn't matter...
    /* IV (organise IV:1-4) */
    /* :1 */
    memCpy(&iv, m_d_xNonleafWorkspace.get(), 1, m_numNodesMinusOne, m_data.numStates());
    /* :2 */
    memCpy(&iv, m_d_uNonleafWorkspace.get(), 1, m_numNodesMinusOne, m_data.numInputs(), m_data.numStates());
    /* :3 */
    memCpy(&iv, &t, 1, m_numNodesMinusOne, 1, m_data.numStatesAndInputs());
    /* :4 */
    memCpy(&iv, &t, 1, m_numNodesMinusOne, 1, m_data.numStatesAndInputs() + 1);
    /* V */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinusOne);
    if (m_data.leafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.leafConstraint()[0]->isRectangle()) {
        xLeaf.deviceCopyTo(v);
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply xLeaf by Gamma_{xN} */
    } else { constraintNotSupported(); }
    /* VI:1 */
    m_d_xLeafWorkspace->addAB(m_data.sqrtStateWeightLeaf(), xLeaf);
    /* VI:2,3 */
    s *= 0.5;  // This affects the current 's'!!! But it shouldn't matter...
    /* VI (organise VI:1-3) */
    /* :1 */
    memCpy(&vi, m_d_xLeafWorkspace.get(), 0, m_numLeafNodesMinusOne, m_data.numStates());
    /* :2 */
    memCpy(&vi, &s, m_tree.numNonleafNodes(), m_numNodesMinusOne, 1, m_data.numStates(), 0, leaf2ZeroLeaf);
    /* :3 */
    memCpy(&vi, &s, m_tree.numNonleafNodes(), m_numNodesMinusOne, 1, m_data.numStates() + 1, 0, leaf2ZeroLeaf);
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {
    /* s (nonleaf) */
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_numNonleafNodesMinusOne);
    ii.deviceCopyTo(sNonleaf);
    /* y */
    i.deviceCopyTo(y);
    y.addAB(m_data.b(), ii, -1., 1.);
    /* x (nonleaf) and u:Gamma */
    if (m_data.nonleafConstraint()[0]->isNone()) {
        DTensor<T> xNonleaf(x, m_matAxis, 0, m_numNonleafNodesMinusOne);
        k_setToZero<<<numBlocks(xNonleaf.numEl(), TPB), TPB>>>(xNonleaf.raw(), xNonleaf.numEl());
        k_setToZero<<<numBlocks(u.numEl(), TPB), TPB>>>(u.raw(), u.numEl());
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
        memCpy(&x, &iii, 0, m_numNonleafNodesMinusOne, m_data.numStates());
        memCpy(&u, &iii, 0, m_numNonleafNodesMinusOne, m_data.numInputs(), 0, m_data.numStates());
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply iii by Gamma^{T}_{x} and Gamma^{T}_{u} */
    } else { constraintNotSupported(); }
    /* x (nonleaf) and u:Weights */
    /* -> Compute `Qiv1` at every nonroot node */
    memCpy(m_d_xNonleafWorkspace.get(), &iv, 1, m_numNodesMinusOne, m_data.numStates());
    m_d_xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *m_d_xNonleafWorkspace);
    /* -> Compute `Riv2` at every nonroot node */
    memCpy(m_d_uNonleafWorkspace.get(), &iv, 1, m_numNodesMinusOne, m_data.numInputs(), 0, m_data.numStates());
    m_d_uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *m_d_uNonleafWorkspace);
    /* -> Add children of each nonleaf node */
    for (size_t chIdx = 0; chIdx < m_tree.numEvents(); chIdx++) {
        /* -> Add to `x` all children `Qiv1` */
        k_memCpyCh2Node<<<m_tree.numNonleafNodes(), TPB>>>(x.raw(), m_d_xNonleafWorkspace->raw(),
                                                           0, m_numNonleafNodesMinusOne, m_data.numStates(), chIdx,
                                                           m_tree.d_childFrom().raw(), m_tree.d_numChildren().raw(),
                                                           true);
        /* -> Add to `u` all children `Riv2` */
        k_memCpyCh2Node<<<m_tree.numNonleafNodes(), TPB>>>(u.raw(), m_d_uNonleafWorkspace->raw(),
                                                           0, m_numNonleafNodesMinusOne, m_data.numInputs(), chIdx,
                                                           m_tree.d_childFrom().raw(), m_tree.d_numChildren().raw(),
                                                           true);
    }
    /* t */
    memCpy(&t, &iv, 1, m_numNodesMinusOne, 1, 0, m_data.numStatesAndInputs());
    memCpy(m_d_scalarWorkspace.get(), &iv, 1, m_numNodesMinusOne, 1, 0, m_data.numStatesAndInputs() + 1);
    t += *m_d_scalarWorkspace;
    t *= 0.5;
    /* x (leaf):Gamma */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinusOne);
    if (m_data.leafConstraint()[0]->isNone()) {
        k_setToZero<<<numBlocks(xLeaf.numEl(), TPB), TPB>>>(xLeaf.raw(), xLeaf.numEl());
    } else if (m_data.leafConstraint()[0]->isRectangle()) {
        v.deviceCopyTo(xLeaf);
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply v by Gamma^{T}_{xN} */
    } else { constraintNotSupported(); }
    /* x (leaf) */
    memCpy(m_d_xLeafWorkspace.get(), &vi, 0, m_numLeafNodesMinusOne, m_data.numStates());
    xLeaf.addAB(m_data.sqrtStateWeightLeaf(), *m_d_xLeafWorkspace, 1., 1.);
    /* s (leaf) */
    memCpy(&s, &vi, m_tree.numNonleafNodes(), m_numNodesMinusOne, 1, 0, m_data.numStates(), zeroLeaf2Leaf);
    memCpy(m_d_scalarWorkspace.get(), &vi, m_tree.numNonleafNodes(), m_numNodesMinusOne, 1, 0, m_data.numStates() + 1,
           zeroLeaf2Leaf);
    DTensor<T> sLeaf(s, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinusOne);
    DTensor<T> wsLeaf(*m_d_scalarWorkspace, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinusOne);
    sLeaf += wsLeaf;
    sLeaf *= 0.5;
}


#endif
