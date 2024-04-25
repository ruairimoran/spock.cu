#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


__global__ void d_projectRectangle(size_t dimension, real_t *vec, real_t *lowerBound, real_t *upperBound);


class Constraint {

protected:
    size_t m_dimension = 0;
    DTensor<real_t> &m_d_vec;

    explicit Constraint(size_t dim, DTensor<real_t> &d_vec) :
        m_dimension(dim), m_d_vec(d_vec) {}

public:
    virtual ~Constraint() {}

    virtual void project() = 0;

    size_t dimension() { return m_dimension; }

};


/**
 * No constraint
 * - used as placeholder
*/
class NoConstraint : public Constraint {

public:
    explicit NoConstraint(size_t dim, DTensor<real_t> &d_vec) :
        Constraint(dim, d_vec) {}

    void project() {
        // Do nothing!
    }

};


/**
 * Rectangle constraint:
 * lb <= x <= ub
 *
 * @param d_vec DTensor under constraint
 * @param lb lower bound
 * @param ub upper bound
*/
class Rectangle : public Constraint {

private:
    DTensor<real_t> &m_d_lowerBound;
    DTensor<real_t> &m_d_upperBound;

public:
    explicit Rectangle(size_t dim,
                       DTensor<real_t> &d_vec,
                       DTensor<real_t> &d_lb,
                       DTensor<real_t> &d_ub) :
        Constraint(dim, d_vec), m_d_lowerBound(d_lb), m_d_upperBound(d_ub) {}

    void project() {
        d_projectRectangle<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(m_dimension, m_d_vec.raw(),
                                                                           m_d_lowerBound.raw(),
                                                                           m_d_upperBound.raw());
    }

};


#endif
