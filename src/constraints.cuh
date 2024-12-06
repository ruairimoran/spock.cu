#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"
#include "memCpy.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t, T *, T *, T *);


/**
 * Constraint types
 */
enum ConstraintMode {
    nonleaf,
    leaf
};


/**
 * Base constraint class
 */
TEMPLATE_WITH_TYPE_T
class Constraint {

protected:
    size_t m_dim = 0;

    explicit Constraint(size_t dim = 0) : m_dim(dim) {}

    bool dimensionCheck(DTensor<T> &d_vec) {
        if (d_vec.numEl() != m_dim) {
            err << "[Constraint] Given DTensor has size (" << d_vec.numEl()
                << "), but constraint has size (" << m_dim << ")\n";
            throw std::invalid_argument(err.str());
        }
        return true;
    }

    virtual std::ostream &print(std::ostream &out) const { return out; };

public:
    virtual ~Constraint() = default;

    virtual size_t dimension() { return m_dim; }

    virtual void constrain(DTensor<T> &) {};

    virtual void op(DTensor<T> &, DTensor<T> &, size_t, size_t, DTensor<T> &, size_t) {};

    virtual void op(DTensor<T> &, DTensor<T> &, size_t, size_t) {};

    virtual void adj(DTensor<T> &, DTensor<T> &, size_t, size_t, DTensor<T> &, size_t) {};

    virtual void adj(DTensor<T> &, DTensor<T> &, size_t, size_t) {};

    friend std::ostream &operator<<(std::ostream &out, const Constraint<T> &data) { return data.print(out); }
};


/**
 * No constraint
 * - used as placeholder
*/
TEMPLATE_WITH_TYPE_T
class NoConstraint : public Constraint<T> {

public:
    explicit NoConstraint() : Constraint<T>() {}

    void adj(DTensor<T> &dual, DTensor<T> &x, size_t nMinusOne, size_t numStates, DTensor<T> &u, size_t numInputs) {
        k_setToZero<<<numBlocks(x.numEl(), TPB), TPB>>>(x.raw(), x.numEl());
        k_setToZero<<<numBlocks(u.numEl(), TPB), TPB>>>(u.raw(), u.numEl());
    }

    void adj(DTensor<T> &dual, DTensor<T> &x, size_t nMinusOne, size_t numStates) {
        k_setToZero<<<numBlocks(x.numEl(), TPB), TPB>>>(x.raw(), x.numEl());
    }
};


/**
 * Rectangle constraint:
 * lb <= x <= ub
 *
 * @param lb lower bound
 * @param ub upper bound
*/
TEMPLATE_WITH_TYPE_T
class Rectangle : public Constraint<T> {

private:
    std::unique_ptr<DTensor<T>> m_d_lowerBound = nullptr;
    std::unique_ptr<DTensor<T>> m_d_upperBound = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Constraint: Rectangle, \n";
        printIfTensor(out, "Lower bound: ", m_d_lowerBound);
        printIfTensor(out, "Upper bound: ", m_d_upperBound);
        return out;
    }

public:
    explicit Rectangle(std::string path, std::string ext) {
        m_d_lowerBound = std::make_unique<DTensor<T>>(DTensor<T>::parseFromFile(path + "LB" + ext));
        m_d_upperBound = std::make_unique<DTensor<T>>(DTensor<T>::parseFromFile(path + "UB" + ext));
        this->m_dim = m_d_lowerBound->numEl();
    }

    void constrain(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_projectRectangle<<<numBlocks(this->m_dim, TPB), TPB>>>(this->m_dim, d_vec.raw(),
                                                                 m_d_lowerBound->raw(), m_d_upperBound->raw());
    }

    void op(DTensor<T> &dual, DTensor<T> &x, size_t nMinusOne, size_t numStates, DTensor<T> &u, size_t numInputs) {
        memCpy(&dual, &x, 0, nMinusOne, numStates);
        memCpy(&dual, &u, 0, nMinusOne, numInputs, numStates);
    }

    void op(DTensor<T> &dual, DTensor<T> &x, size_t nMinusOne, size_t numStates) {
        memCpy(&dual, &x, 0, nMinusOne, numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x, size_t nMinusOne, size_t numStates, DTensor<T> &u, size_t numInputs) {
        memCpy(&x, &dual, 0, nMinusOne, numStates);
        memCpy(&u, &dual, 0, nMinusOne, numInputs, 0, numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x, size_t nMinusOne, size_t numStates) {
        memCpy(&x, &dual, 0, nMinusOne, numStates);
    }
};


/**
 * Ball constraint
 * ||x||_{2} <= ub
 *
 * @param ub upper bound
*/
TEMPLATE_WITH_TYPE_T
class Ball : public Constraint<T> {

public:
    explicit Ball(size_t dim) : Constraint<T>(dim) {}

};


#endif
