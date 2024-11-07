#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include <chrono>


TEMPLATE_WITH_TYPE_T
__global__ void k_setToZero(T *, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpy(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpy(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
void memCpy(DTensor<T> *dst, DTensor<T> *src,
            size_t nodeFrom, size_t nodeTo, size_t numEl,
            size_t elFromDst = 0, size_t elFromSrc = 0,
            DTensor<size_t> *ancestors = nullptr) {
    size_t nodeSizeDst = dst->numRows();
    size_t nodeSizeSrc = src->numRows();
    if (dst->numCols() != 1 || src->numCols() != 1) throw std::invalid_argument("[memCpy] numCols must be 1.");
    if (std::max(nodeSizeDst, nodeSizeSrc) > TPB) throw std::invalid_argument("[memCpy] Node data too large.");
    if (ancestors && nodeFrom < 1) throw std::invalid_argument("[memCpy] Root node has no ancestor.");
    size_t nBlocks = nodeTo + 1;
    if (ancestors) {
        k_memCpy<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst, nodeSizeSrc,
                                   elFromDst, elFromSrc, ancestors->raw());
    } else {
        k_memCpy<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst, nodeSizeSrc,
                                   elFromDst, elFromSrc);
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
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_tree.numNonleafNodes() - 1);
    sNonleaf.deviceCopyTo(ii);
    ii.addAB(m_data.bTr(), y, -1., 1.);
    /* III */
    if (m_data.nonleafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
        memCpy(&iii, &x, 0, m_tree.numNonleafNodes() - 1, m_data.numStates());
        memCpy(&iii, &u, 0, m_tree.numNonleafNodes() - 1, m_data.numInputs(), m_data.numStates());
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply xuNonleaf by Gamma_{xu} */
    } else { constraintNotSupported(); }
    /* IV:1 */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        size_t anc = m_tree.ancestors()[node];
        DTensor<T> xAnc(x, m_matAxis, anc, anc);
        DTensor<T> xNode(*m_d_xNonleafWorkspace, m_matAxis, node, node);
        xAnc.deviceCopyTo(xNode);
    }
    m_d_xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *m_d_xNonleafWorkspace);
    /* IV:2 */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        size_t anc = m_tree.ancestors()[node];
        DTensor<T> uAnc(u, m_matAxis, anc, anc);
        DTensor<T> uNode(*m_d_uNonleafWorkspace, m_matAxis, node, node);
        uAnc.deviceCopyTo(uNode);
    }
    m_d_uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *m_d_uNonleafWorkspace);
    /* IV:3,4 */
    t *= 0.5;  // This affects the current 't'!!! But it shouldn't matter...
    /* IV (organise IV:1-4) */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        DTensor<T> ivNode(iv, m_matAxis, node, node);
        DTensor<T> tNode(t, m_matAxis, node, node);
        /* :1 */
        DTensor<T> xNode(*m_d_xNonleafWorkspace, m_matAxis, node, node);
        DTensor<T> iv1(ivNode, 0, 0, m_data.numStates() - 1);
        xNode.deviceCopyTo(iv1);
        /* :2 */
        DTensor<T> uNode(*m_d_uNonleafWorkspace, m_matAxis, node, node);
        DTensor<T> iv2(ivNode, 0, m_data.numStates(), m_data.numStatesAndInputs() - 1);
        uNode.deviceCopyTo(iv2);
        /* :3 */
        DTensor<T> iv3(ivNode, 0, m_data.numStatesAndInputs(), m_data.numStatesAndInputs());
        tNode.deviceCopyTo(iv3);
        /* :4 */
        DTensor<T> iv4(ivNode, 0, m_data.numStatesAndInputs() + 1, m_data.numStatesAndInputs() + 1);
        tNode.deviceCopyTo(iv4);
    }
    /* V */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
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
    for (size_t node = m_tree.numNonleafNodes(); node < m_tree.numNodes(); node++) {
        size_t idx = node - m_tree.numNonleafNodes();
        DTensor<T> viNode(vi, m_matAxis, idx, idx);
        DTensor<T> sNode(s, m_matAxis, node, node);
        /* :1 */
        DTensor<T> xNode(*m_d_xLeafWorkspace, m_matAxis, idx, idx);
        DTensor<T> vi1(viNode, 0, 0, m_data.numStates() - 1);
        xNode.deviceCopyTo(vi1);
        /* :2 */
        DTensor<T> vi2(viNode, 0, m_data.numStates(), m_data.numStates());
        sNode.deviceCopyTo(vi2);
        /* :3 */
        DTensor<T> vi3(viNode, 0, m_data.numStates() + 1, m_data.numStates() + 1);
        sNode.deviceCopyTo(vi3);
    }
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {
    /* s (nonleaf) */
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_tree.numNonleafNodes() - 1);
    ii.deviceCopyTo(sNonleaf);
    /* y */
    i.deviceCopyTo(y);
    y.addAB(m_data.b(), ii, -1., 1.);
    /* x (nonleaf) and u:Gamma */
    if (m_data.nonleafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.nonleafConstraint()[0]->isRectangle()) {
        for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
            DTensor<T> iiiNode(iii, m_matAxis, node, node);
            DTensor<T> iiiX(iiiNode, 0, 0, m_data.numStates() - 1);
            DTensor<T> iiiU(iiiNode, 0, m_data.numStates(), m_data.numStatesAndInputs() - 1);
            DTensor<T> xNode(x, m_matAxis, node, node);
            DTensor<T> uNode(u, m_matAxis, node, node);
            iiiX.deviceCopyTo(xNode);
            iiiU.deviceCopyTo(uNode);
        }
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply iii by Gamma^{T}_{x} and Gamma^{T}_{u} */
    } else { constraintNotSupported(); }
    /* x (nonleaf) */
    /* -> Compute `Qiv1` at every nonroot node */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        DTensor<T> ivNode(iv, m_matAxis, node, node);
        DTensor<T> iv1(ivNode, 0, 0, m_data.numStates() - 1);
        DTensor<T> ws(*m_d_xNonleafWorkspace, m_matAxis, node, node);
        iv1.deviceCopyTo(ws);
    }
    m_d_xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *m_d_xNonleafWorkspace);
    /* -> Add to `x` all children `Qiv1` of each nonleaf node (zeros added if child does not exist) */
    DTensor<T> x_Nonleaf(x, m_matAxis, 0, m_tree.numNonleafNodes() - 1);
    for (size_t chIdx = 0; chIdx < m_tree.numEvents(); chIdx++) {  // Index of child of every nonleaf node
        for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
            size_t ch = m_tree.childFrom()[node] + chIdx;
            DTensor<T> Qiv1_Node(*m_d_xNonleafAdd, m_matAxis, node, node);
            if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Qiv1`
                DTensor<T> Qiv1_ChNode(*m_d_xNonleafWorkspace, m_matAxis, ch, ch);
                Qiv1_ChNode.deviceCopyTo(Qiv1_Node);
            } else {  // If child does not exist, copy in zeros
                k_setToZero<<<numBlocks(m_data.numStates(), TPB), TPB>>>(Qiv1_Node.raw(), m_data.numStates());
            }
        }
        x_Nonleaf += *m_d_xNonleafAdd;
    }
    /* u */
    /* -> Compute `Riv2` at every nonroot node */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        DTensor<T> ivNode(iv, m_matAxis, node, node);
        DTensor<T> iv2(ivNode, 0, m_data.numStates(), m_data.numStatesAndInputs() - 1);
        DTensor<T> ws(*m_d_uNonleafWorkspace, m_matAxis, node, node);
        iv2.deviceCopyTo(ws);
    }
    m_d_uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *m_d_uNonleafWorkspace);
    /* -> Add to `u` all children `Riv2` of each nonleaf node (zeros added if child does not exist) */
    for (size_t chIdx = 0; chIdx < m_tree.numEvents(); chIdx++) {  // Index of child of every nonleaf node
        for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
            size_t ch = m_tree.childFrom()[node] + chIdx;
            DTensor<T> Riv2_Node(*m_d_uNonleafAdd, m_matAxis, node, node);
            if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Qiv1`
                DTensor<T> Riv2_ChNode(*m_d_uNonleafWorkspace, m_matAxis, ch, ch);
                Riv2_ChNode.deviceCopyTo(Riv2_Node);
            } else {  // If child does not exist, copy in zeros
                k_setToZero<<<numBlocks(m_data.numInputs(), TPB), TPB>>>(Riv2_Node.raw(), m_data.numInputs());
            }
        }
        u += *m_d_uNonleafAdd;
    }
    /* t */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        DTensor<T> ivNode(iv, m_matAxis, node, node);
        DTensor<T> tNode(t, m_matAxis, node, node);
        DTensor<T> wsNode(*m_d_scalarWorkspace, m_matAxis, node, node);
        DTensor<T> iv3(ivNode, 0, m_data.numStatesAndInputs(), m_data.numStatesAndInputs());
        iv3.deviceCopyTo(tNode);
        DTensor<T> iv4(ivNode, 0, m_data.numStatesAndInputs() + 1, m_data.numStatesAndInputs() + 1);
        iv4.deviceCopyTo(wsNode);
    }
    t += *m_d_scalarWorkspace;
    t *= 0.5;
    /* x (leaf):Gamma */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    if (m_data.leafConstraint()[0]->isNone()) {
        /* Do nothing */
    } else if (m_data.leafConstraint()[0]->isRectangle()) {
        v.deviceCopyTo(xLeaf);
    } else if (m_data.leafConstraint()[0]->isBall()) {
        /* TODO! Pre-multiply v by Gamma^{T}_{xN} */
    } else { constraintNotSupported(); }
    /* x (leaf) */
    for (size_t idx = 0; idx < m_tree.numLeafNodes(); idx++) {
        DTensor<T> viNode(vi, m_matAxis, idx, idx);
        DTensor<T> vi1(viNode, 0, 0, m_data.numStates() - 1);
        DTensor<T> ws(*m_d_xLeafWorkspace, m_matAxis, idx, idx);
        vi1.deviceCopyTo(ws);
    }
    xLeaf.addAB(m_data.sqrtStateWeightLeaf(), *m_d_xLeafWorkspace, 1., 1.);
    /* s (leaf) */
    for (size_t node = m_tree.numNonleafNodes(); node < m_tree.numNodes(); node++) {
        size_t idx = node - m_tree.numNonleafNodes();
        DTensor<T> viNode(vi, m_matAxis, idx, idx);
        DTensor<T> sNode(s, m_matAxis, node, node);
        DTensor<T> wsNode(*m_d_scalarWorkspace, m_matAxis, node, node);
        DTensor<T> vi2(viNode, 0, m_data.numStates(), m_data.numStates());
        vi2.deviceCopyTo(sNode);
        DTensor<T> vi3(viNode, 0, m_data.numStates() + 1, m_data.numStates() + 1);
        vi3.deviceCopyTo(wsNode);
    }
    DTensor<T> sLeaf(*m_d_scalarWorkspace, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    DTensor<T> wsLeaf(*m_d_scalarWorkspace, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    sLeaf += wsLeaf;
    sLeaf *= 0.5;
}


#endif
