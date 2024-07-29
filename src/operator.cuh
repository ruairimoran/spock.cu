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
    std::unique_ptr<DTensor<T>> uNonleafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> xNonleafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> xLeafWorkspace = nullptr;

public:
    /**
     * Constructor
     */
    LinearOperator(ScenarioTree<T> &tree, ProblemData<T> &data) :
        m_tree(tree), m_data(data) {
        uNonleafWorkspace = std::make_unique<DTensor<T>>(m_data.numInputs(), 1, m_tree.numNodes(), true);
        xNonleafWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes(), true);
        xLeafWorkspace = std::make_unique<DTensor<T>>(m_data.numStates(), 1, m_tree.numNodes() - m_tree.numNonleafNodes(), true);
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
    ii.addAB(m_data.bTr(), y, -1.);
    /* III */
    if (m_data.nonleafConstraint()[0]->isBall()) { /* Pre-multiply xuNonleaf by Gamma_{xu} */ }
    /* IV:1 */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        size_t anc = m_tree.ancestors()[node];
        DTensor<T> xAnc(x, m_matAxis, anc, anc);
        DTensor<T> xNode(*xNonleafWorkspace, m_matAxis, node, node);
        xAnc.deviceCopyTo(xNode);
    }
    xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *xNonleafWorkspace);
    /* IV:2 */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        size_t anc = m_tree.ancestors()[node];
        DTensor<T> uAnc(u, m_matAxis, anc, anc);
        DTensor<T> uNode(*uNonleafWorkspace, m_matAxis, node, node);
        uAnc.deviceCopyTo(uNode);
    }
    uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *uNonleafWorkspace);
    /* IV (organise IV:1-4) */
    for (size_t node = 1; node < m_tree.numNodes(); node++) {
        DTensor<T> ivNode(iv, m_matAxis, node, node);
        DTensor<T> tNode(t, m_matAxis, node, node);
        tNode *= 0.5;  // This affects the current 't'!!! But it shouldn't matter...
        /* :1 */
        DTensor<T> xNode(*xNonleafWorkspace, m_matAxis, node, node);
        DTensor<T> iv1(ivNode, 0, 0, m_data.numStates() - 1);
        xNode.deviceCopyTo(iv1);
        /* :2 */
        DTensor<T> uNode(*uNonleafWorkspace, m_matAxis, node, node);
        DTensor<T> iv2(ivNode, 0, m_data.numStates(), m_data.numStatesAndInputs() - 1);
        uNode.deviceCopyTo(iv2);
        /* IV:3 */
        DTensor<T> iv3(ivNode, 0, m_data.numStatesAndInputs(), m_data.numStatesAndInputs());
        tNode.deviceCopyTo(iv3);
        /* IV:4 */
        DTensor<T> iv4(ivNode, 0, m_data.numStatesAndInputs() + 1, m_data.numStatesAndInputs() + 1);
        tNode.deviceCopyTo(iv4);
    }
    /* V */
    if (m_data.leafConstraint()[m_tree.numNonleafNodes()]->isBall()) { /* Pre-multiply xLeaf by Gamma_{xN} */ }
    /* VI:1 */
    DTensor<T> sqrtQLeaf(m_data.sqrtStateWeightLeaf(), m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodes() - 1);
    xLeafWorkspace->addAB(sqrtQLeaf, xLeaf);
    /* VI (organise VI:1-3) */
    for (size_t node = m_tree.numNonleafNodes(); node < m_tree.numNodes(); node++) {
        size_t idx = node - m_tree.numNonleafNodes();
        DTensor<T> viNode(iv, m_matAxis, node, node);
        DTensor<T> sNode(s, m_matAxis, node, node);
        sNode *= 0.5;  // This affects the current 's'!!! But it shouldn't matter...
        /* VI:1 */
        DTensor<T> xNode(*xLeafWorkspace, m_matAxis, idx, idx);
        DTensor<T> vi1(viNode, 0, 0, m_data.numStates() - 1);
        xNode.deviceCopyTo(vi1);
        /* VI:2 */
        DTensor<T> vi2(viNode, 0, m_data.numStates(), m_data.numStates());
        sNode.deviceCopyTo(vi2);
        /* VI:3 */
        DTensor<T> vi3(viNode, 0, m_data.numStates() + 1, m_data.numStates() + 1);
        sNode.deviceCopyTo(vi3);
    }
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {

}


#endif
