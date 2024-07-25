#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include "../include/gpu.cuh"



/**
 * Linear operator 'L' and its adjoint
 */
TEMPLATE_WITH_TYPE_T
class LinearOperator {
public:
    /**
     * Constructor
     */
    LinearOperator() {}

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
    y.deviceCopyTo(ii);
    ii.dotF()
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {

}


#endif
