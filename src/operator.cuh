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
    size_t m_matAxis = 2;

public:
    /**
     * Constructor
     */
    LinearOperator(ScenarioTree<T> &tree, ProblemData<T> &data) : m_tree(tree), m_data(data) {}

    ~LinearOperator() = default;

    /**
     * For reuse while testing.
     */
    void resetWorkspace() {};

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
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_tree.numNonleafNodesMinus1());
    sNonleaf.deviceCopyTo(ii);
    ii.addAB(m_data.risk()->bTr(), y, -1., 1.);
    /* III */
    m_data.nonleafConstraint()->op(iii, x, u);
    /* IV */
    m_data.nonleafCost()->op(iv, x, u, t);
    /* V */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodesMinus1());
    m_data.leafConstraint()->op(v, xLeaf);
    /* VI */
    DTensor<T> sLeaf(s, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodesMinus1());
    m_data.leafCost()->op(vi, xLeaf, sLeaf);
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {
    /* s (nonleaf) */
    DTensor<T> sNonleaf(s, m_matAxis, 0, m_tree.numNonleafNodesMinus1());
    ii.deviceCopyTo(sNonleaf);
    /* y */
    i.deviceCopyTo(y);
    y.addAB(m_data.risk()->b(), ii, -1., 1.);
    /* x (nonleaf) and u:Gamma */
    m_data.nonleafConstraint()->adj(iii, x, u);
    /* x (nonleaf) and u:Weights */
    m_data.nonleafCost()->adj(iv, x, u, t);
    /* x (leaf):Gamma */
    DTensor<T> xLeaf(x, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodesMinus1());
    m_data.leafConstraint()->adj(v, xLeaf);
    /* x (leaf) and s (leaf) */
    DTensor<T> sLeaf(s, m_matAxis, m_tree.numNonleafNodes(), m_tree.numNodesMinus1());
    m_data.leafCost()->adj(vi, xLeaf, sLeaf);
}


#endif
