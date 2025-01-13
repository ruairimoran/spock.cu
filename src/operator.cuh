#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "problem.cuh"
#include <chrono>


/**
 * Linear operator 'L' and its adjoint
 */
TEMPLATE_WITH_TYPE_T
class LinearOperator {
protected:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
    size_t m_numNodesMinus1 = m_tree.numNodesMinus1();
    size_t m_numNonleafNodesMinus1 = m_tree.numNonleafNodesMinus1();
    size_t m_numLeafNodesMinus1 = m_tree.numLeafNodesMinus1();
    size_t m_matAxis = 2;
    std::unique_ptr<DTensor<T>> m_d_uNonleafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xNonleafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xLeafWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_scalarWorkspace = nullptr;

public:
    /**
     * Constructor
     */
    LinearOperator(ScenarioTree<T> &tree, ProblemData<T> &data) : m_tree(tree), m_data(data) {
        m_d_uNonleafWorkspace = std::make_unique<DTensor<T>>(m_tree.numInputs(), 1, m_tree.numNodes(), true);
        m_d_xNonleafWorkspace = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, m_tree.numNodes(), true);
        m_d_xLeafWorkspace = std::make_unique<DTensor<T>>(m_tree.numStates(), 1, m_tree.numLeafNodes(), true);
        m_d_scalarWorkspace = std::make_unique<DTensor<T>>(1, 1, m_tree.numNodes(), true);
    }

    ~LinearOperator() = default;

    /**
     * For reuse while testing.
     */
    void resetWorkspace() {
        m_d_uNonleafWorkspace->upload(std::vector<T>(m_tree.numInputs() * m_tree.numNodes(), 0));
        m_d_xNonleafWorkspace->upload(std::vector<T>(m_tree.numStates() * m_tree.numNodes(), 0));
        m_d_xLeafWorkspace->upload(std::vector<T>(m_tree.numStates() * m_tree.numLeafNodes(), 0));
        m_d_scalarWorkspace->upload(std::vector<T>(m_tree.numNodes(), 0));
    };

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
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_numNonleafNodesMinus1);
    sNonleaf.deviceCopyTo(ii);
    ii.addAB(m_data.risk()->bTr(), y, -1., 1.);
    /* III */
    m_data.nonleafConstraint()->op(iii, x, u);
    /* IV:1 */
    m_tree.memCpyAnc2Node(*m_d_xNonleafWorkspace, x, 1, m_numNodesMinus1, m_tree.numStates(), 0, 0);
    m_d_xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *m_d_xNonleafWorkspace);
    /* IV:2 */
    m_tree.memCpyAnc2Node(*m_d_uNonleafWorkspace, u, 1, m_numNodesMinus1, m_tree.numInputs(), 0, 0);
    m_d_uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *m_d_uNonleafWorkspace);
    /* IV:3,4 */
    t *= 0.5;  // This affects the current 't'!!! But it shouldn't matter...
    /* IV (organise IV:1-4) */
    /* :1 */
    memCpyNode2Node(iv, *m_d_xNonleafWorkspace, 1, m_numNodesMinus1, m_tree.numStates());
    /* :2 */
    memCpyNode2Node(iv, *m_d_uNonleafWorkspace, 1, m_numNodesMinus1, m_tree.numInputs(), m_tree.numStates());
    /* :3 */
    memCpyNode2Node(iv, t, 1, m_numNodesMinus1, 1, m_tree.numStatesAndInputs());
    /* :4 */
    memCpyNode2Node(iv, t, 1, m_numNodesMinus1, 1, m_tree.numStatesAndInputs() + 1);
    /* V */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinus1);
    m_data.leafConstraint()->op(v, xLeaf);
    /* VI:1 */
    m_d_xLeafWorkspace->addAB(m_data.sqrtStateWeightLeaf(), xLeaf);
    /* VI:2,3 */
    s *= 0.5;  // This affects the current 's'!!! But it shouldn't matter...
    /* VI (organise VI:1-3) */
    /* :1 */
    memCpyNode2Node(vi, *m_d_xLeafWorkspace, 0, m_numLeafNodesMinus1, m_tree.numStates());
    /* :2 */
    m_tree.memCpyLeaf2Zero(vi, s, 1, m_tree.numStates(), 0);
    /* :3 */
    m_tree.memCpyLeaf2Zero(vi, s, 1, m_tree.numStates() + 1, 0);
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {
    /* s (nonleaf) */
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_numNonleafNodesMinus1);
    ii.deviceCopyTo(sNonleaf);
    /* y */
    i.deviceCopyTo(y);
    y.addAB(m_data.risk()->b(), ii, -1., 1.);
    /* x (nonleaf) and u:Gamma */
    m_data.nonleafConstraint()->adj(iii, x, u);
    /* x (nonleaf) and u:Weights */
    /* -> Compute `Qiv1` at every nonroot node */
    memCpyNode2Node(*m_d_xNonleafWorkspace, iv, 1, m_numNodesMinus1, m_tree.numStates());
    m_d_xNonleafWorkspace->addAB(m_data.sqrtStateWeight(), *m_d_xNonleafWorkspace);
    /* -> Compute `Riv2` at every nonroot node */
    memCpyNode2Node(*m_d_uNonleafWorkspace, iv, 1, m_numNodesMinus1, m_tree.numInputs(), 0,
                           m_tree.numStates());
    m_d_uNonleafWorkspace->addAB(m_data.sqrtInputWeight(), *m_d_uNonleafWorkspace);
    /* -> Add children of each nonleaf node */
    for (size_t chIdx = 0; chIdx < m_tree.numEvents(); chIdx++) {
        /* -> Add to `x` all children `Qiv1` */
        m_tree.memCpyCh2Node(x, *m_d_xNonleafWorkspace, 0, m_numNonleafNodesMinus1, chIdx, true);
        /* -> Add to `u` all children `Riv2` */
        m_tree.memCpyCh2Node(u, *m_d_uNonleafWorkspace, 0, m_numNonleafNodesMinus1, chIdx, true);
    }
    /* t */
    memCpyNode2Node(t, iv, 1, m_numNodesMinus1, 1, 0, m_tree.numStatesAndInputs());
    memCpyNode2Node(*m_d_scalarWorkspace, iv, 1, m_numNodesMinus1, 1, 0, m_tree.numStatesAndInputs() + 1);
    t += *m_d_scalarWorkspace;
    t *= 0.5;
    /* x (leaf):Gamma */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinus1);
    m_data.leafConstraint()->adj(v, xLeaf);
    /* x (leaf) */
    memCpyNode2Node(*m_d_xLeafWorkspace, vi, 0, m_numLeafNodesMinus1, m_tree.numStates());
    xLeaf.addAB(m_data.sqrtStateWeightLeaf(), *m_d_xLeafWorkspace, 1., 1.);
    /* s (leaf) */
    m_tree.memCpyZero2Leaf(s, vi, 1, 0, m_tree.numStates());
    m_tree.memCpyZero2Leaf(*m_d_scalarWorkspace, vi, 1, 0, m_tree.numStates() + 1);
    DTensor<T> sLeaf(s, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinus1);
    DTensor<T> wsLeaf(*m_d_scalarWorkspace, m_matAxis, m_tree.numNonleafNodes(), m_numNodesMinus1);
    sLeaf += wsLeaf;
    sLeaf *= 0.5;
}


#endif
