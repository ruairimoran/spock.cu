#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t, T *, T *, T *);


TEMPLATE_WITH_TYPE_T
class Constraint {

protected:
    size_t m_nodeIndex = 0;
    size_t m_dimension = 0;
    std::unique_ptr<DTensor<T>> m_d_empty = nullptr;

    explicit Constraint(size_t node, size_t dim) : m_nodeIndex(node), m_dimension(dim) {
        m_d_empty = std::make_unique<DTensor<T>>(0);
    }

    bool dimensionCheck(DTensor<T> &d_vec) {
        if (d_vec.numRows() != m_dimension || d_vec.numCols() != 1 || d_vec.numMats() != 1) {
            err << "DTensor is [" << d_vec.numRows() << " x " << d_vec.numCols() << " x " << d_vec.numMats()
                << "], but constraint has dimensions [" << m_dimension << " x " << 1 << " x " << 1 << "]\n";
            throw std::invalid_argument(err.str());
        }
        return true;
    }

public:
    virtual ~Constraint() = default;

    virtual size_t node() { return m_nodeIndex; }

    virtual size_t dimension() { return m_dimension; }

    virtual void project(DTensor<T> &d_vec) {};

    virtual bool isNone() { return false; }

    virtual bool isRectangle() { return false; }

    virtual DTensor<T> &lo() { return *m_d_empty; }

    virtual DTensor<T> &hi() { return *m_d_empty; }

    virtual bool isBall() { return false; }

    virtual void print() {};
};


/**
 * No constraint
 * - used as placeholder
*/
TEMPLATE_WITH_TYPE_T
class NoConstraint : public Constraint<T> {

public:
    explicit NoConstraint(size_t node, size_t dim) : Constraint<T>(node, dim) {}

    bool isNone() { return true; }
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

public:
    explicit Rectangle(size_t node, size_t dim, std::vector<T> &lb, std::vector<T> &ub) : Constraint<T>(node, dim) {
        m_d_lowerBound = std::make_unique<DTensor<T>>(lb, dim);
        m_d_upperBound = std::make_unique<DTensor<T>>(ub, dim);
    }

    void project(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_projectRectangle<<<numBlocks(this->m_dimension, TPB), TPB>>>(this->m_dimension, d_vec.raw(),
                                                                       m_d_lowerBound->raw(), m_d_upperBound->raw());
    }

    bool isRectangle() { return true; }

    DTensor<T> &lo() { return *m_d_lowerBound; }

    DTensor<T> &hi() { return *m_d_upperBound; }

    void print() {
        std::cout << "Node: " << this->m_nodeIndex << ", Constraint: Rectangle, \n";
        printIfTensor("Lower bound: ", m_d_lowerBound);
        printIfTensor("Upper bound: ", m_d_upperBound);
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
    explicit Ball(size_t node, size_t dim) : Constraint<T>(node, dim) {}

    bool isBall() { return true; }
};


#endif
