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

public:
    /**
     * Constructor
     */
    LinearOperator(ScenarioTree<T> &tree, ProblemData<T> &data) :
        m_tree(tree), m_data(data) {}

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
    if (m_data.nonleafConstraint()[0]->isBall()) { /* Multiply xuNonleaf by Gamma_{xu} */ }
    if (m_data.leafConstraint()[m_tree.numNonleafNodes()]->isBall()) { /* Multiply xLeaf by Gamma_{xN} */ }
    /* IV */

}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {

}


#endif
