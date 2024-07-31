#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include "../include/gpu.cuh"



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
    if (m_data.nonleafConstraint()[0]->isNone() || m_data.nonleafConstraint()[0]->isRectangle()) {
        for (size_t node = 0; node < m_tree.numNonleafNodes(); node++) {
            DTensor<T> iiiNode(iii, m_matAxis, node, node);
            DTensor<T> iiiX(iiiNode, 0, 0, m_data.numStates() - 1);
            DTensor<T> iiiU(iiiNode, 0, m_data.numStates(), m_data.numStatesAndInputs() - 1);
            DTensor<T> xNode(x, m_matAxis, node, node);
            DTensor<T> uNode(u, m_matAxis, node, node);
            xNode.deviceCopyTo(iiiX);
            uNode.deviceCopyTo(iiiU);
        }
    } else if (m_data.nonleafConstraint()[0]->isBall()) {
        /* Pre-multiply xuNonleaf by Gamma_{xu} */
    } else {
        throw std::invalid_argument("[L operator] nonleaf constraint given is not supported.");
    }
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
    if (m_data.leafConstraint()[m_tree.numNonleafNodes()]->isNone() || m_data.leafConstraint()[m_tree.numNonleafNodes()]->isRectangle()) {
        DTensor<T> vLeaf(v, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
        xLeaf.deviceCopyTo(vLeaf);
    } else if (m_data.leafConstraint()[m_tree.numNonleafNodes()]->isBall()) {
        /* Pre-multiply xLeaf by Gamma_{xN} */
    } else {
        throw std::invalid_argument("[L operator] leaf constraint given is not supported.");
    }
    /* VI:1 */
    m_d_xLeafWorkspace->addAB(m_data.sqrtStateWeightLeaf(), xLeaf);
    /* VI:2,3 */
    s *= 0.5;  // This affects the current 's'!!! But it shouldn't matter...
    /* VI (organise VI:1-3) */
    for (size_t node = m_tree.numNonleafNodes(); node < m_tree.numNodes(); node++) {
        size_t idx = node - m_tree.numNonleafNodes();
        DTensor<T> viNode(vi, m_matAxis, node, node);
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
    if (m_data.nonleafConstraint()[0]->isNone() || m_data.nonleafConstraint()[0]->isRectangle()) {
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
        /* Pre-multiply iii by Gamma^{T}_{x} and Gamma^{T}_{u} */
    } else {
        throw std::invalid_argument("[L adjoint] nonleaf constraint given is not supported.");
    }
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
            if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Qiv1`
                DTensor<T> Qiv1_ChNode(*m_d_xNonleafWorkspace, m_matAxis, ch, ch);
                DTensor<T> Qiv1_Node(*m_d_xNonleafWorkspace, m_matAxis, node, node);
                Qiv1_ChNode.deviceCopyTo(Qiv1_Node);
            }
        }
        DTensor<T> x_Add(*m_d_xNonleafWorkspace, m_matAxis, 0, m_tree.numNonleafNodes() - 1);
        x_Nonleaf += x_Add;
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
            if (ch <= m_tree.childTo()[node]) {  // If more children exist, copy in their `Qiv1`
                DTensor<T> Riv2_ChNode(*m_d_uNonleafWorkspace, m_matAxis, ch, ch);
                DTensor<T> Riv2_Node(*m_d_uNonleafWorkspace, m_matAxis, node, node);
                Riv2_ChNode.deviceCopyTo(Riv2_Node);
            }
        }
        u += *m_d_uNonleafWorkspace;
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
    if (m_data.leafConstraint()[m_tree.numNonleafNodes()]->isNone() || m_data.leafConstraint()[m_tree.numNonleafNodes()]->isRectangle()) {
        DTensor<T> vLeaf(v, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
        vLeaf.deviceCopyTo(xLeaf);
    } else if (m_data.leafConstraint()[m_tree.numNonleafNodes()]->isBall()) {
        /* Pre-multiply v by Gamma^{T}_{xN} */
    } else {
        throw std::invalid_argument("[L adjoint] leaf constraint given is not supported.");
    }
    /* x (leaf) */
    for (size_t node = m_tree.numNonleafNodes(); node < m_tree.numNodes(); node++) {
        size_t idx = node - m_tree.numNonleafNodes();
        DTensor<T> viNode(vi, m_matAxis, node, node);
        DTensor<T> vi1(viNode, 0, 0, m_data.numStates() - 1);
        DTensor<T> ws(*m_d_xLeafWorkspace, m_matAxis, idx, idx);
        vi1.deviceCopyTo(ws);
    }
    xLeaf.addAB(m_data.sqrtStateWeightLeaf(), *m_d_xLeafWorkspace, 1., 1.);
    /* s (leaf) */
    for (size_t node = m_tree.numNonleafNodes(); node < m_tree.numNodes(); node++) {
        DTensor<T> viNode(vi, m_matAxis, node, node);
        DTensor<T> sNode(s, m_matAxis, node, node);
        DTensor<T> wsNode(*m_d_scalarWorkspace, m_matAxis, node, node);
        DTensor<T> vi2(viNode, 0, m_data.numStates(), m_data.numStates());
        vi2.deviceCopyTo(sNode);
        DTensor<T> vi3(viNode, 0, m_data.numStates() + 1, m_data.numStates() + 1);
        vi3.deviceCopyTo(wsNode);
    }
    DTensor<T> sLeaf(*m_d_scalarWorkspace, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    DTensor<T> wsLeaf(*m_d_scalarWorkspace, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    sLeaf += *m_d_scalarWorkspace;
    sLeaf *= 0.5;
}


#endif
