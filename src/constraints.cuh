#ifndef __CONSTRAINTS__
#define __CONSTRAINTS__

#include "../include/stdgpu.h"
#include "cones.cuh"


__global__ void d_projectRectangle(size_t dimension, real_t *vec, real_t *lowerBound, real_t *upperBound);


class Constraint {

protected:
    Context &m_context;
    size_t m_dimension = 0;
    DeviceVector<real_t> &m_d_vec;

    explicit Constraint(Context &context, size_t dim, DeviceVector<real_t> &d_vec) :
            m_context(context), m_dimension(dim), m_d_vec(d_vec) {}

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
    explicit NoConstraint(Context &context, size_t dim, DeviceVector<real_t> &d_vec) : Constraint(context, dim,
                                                                                                  d_vec) {}

    void project() {
        // Do nothing!
    }

};


/**
 * Rectangle constraint:
 * lb <= x <= ub
 *
 * @param d_vec DeviceVector under constraint
 * @param lb lower bound
 * @param ub upper bound
*/
class Rectangle : public Constraint {

private:
    DeviceVector<real_t> &m_d_lowerBound;
    DeviceVector<real_t> &m_d_upperBound;

public:
    explicit Rectangle(Context &context, size_t dim,
                       DeviceVector<real_t> &d_vec,
                       DeviceVector<real_t> &d_lb,
                       DeviceVector<real_t> &d_ub) :
            Constraint(context, dim, d_vec), m_d_lowerBound(d_lb), m_d_upperBound(d_ub) {}

    void project() {
        d_projectRectangle<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(m_dimension, m_d_vec.get(),
                                                                           m_d_lowerBound.get(),
                                                                           m_d_upperBound.get());
    }

};


#endif
