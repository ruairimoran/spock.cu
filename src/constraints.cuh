#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


__global__ void d_projectRectangle(size_t dimension, real_t *vec, real_t *lowerBound, real_t *upperBound);


template<typename T>
class Constraint {

protected:
    size_t m_nodeIndex = 0;
    size_t m_dimension = 0;

    explicit Constraint(size_t node, size_t dim) : m_nodeIndex(node), m_dimension(dim) {}

    bool dimensionCheck(DTensor<T> &d_vec) {
        if (d_vec.numRows() != m_dimension || d_vec.numCols() != 1 || d_vec.numMats() != 1) {
            std::cerr << "DTensor is [" << d_vec.numRows() << " x " << d_vec.numCols() << " x " << d_vec.numMats()
                      << "], but constraint has dimensions [" << m_dimension << " x " << 1 << " x " << 1 << "]\n";
            throw std::invalid_argument("DTensor and constraint dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~Constraint() {}

    size_t node() { return m_nodeIndex; }

    size_t dimension() { return m_dimension; }

    virtual void project(DTensor<T> &d_vec) = 0;

    virtual void print() = 0;
};


/**
 * No constraint
 * - used as placeholder
*/
class NoConstraint : public Constraint<void> {

public:
    explicit NoConstraint(size_t node, size_t dim) : Constraint<void>(node, dim) {}
};


/**
 * Rectangle constraint:
 * lb <= x <= ub
 *
 * @param lb lower bound
 * @param ub upper bound
*/
template<typename T>
class Rectangle : public Constraint<T> {

private:
    std::unique_ptr<DTensor<T>> m_d_lowerBound;
    std::unique_ptr<DTensor<T>> m_d_upperBound;

public:
    explicit Rectangle(size_t node, size_t dim, std::vector<T> &lb, std::vector<T> &ub) : Constraint<T>(node, dim) {
        m_d_lowerBound = std::make_unique<DTensor<T>>(lb, dim);
        m_d_upperBound = std::make_unique<DTensor<T>>(ub, dim);
    }

    void project(DTensor<T> &d_vec) {
        Constraint<T>::dimensionCheck(d_vec);
        d_projectRectangle<<<DIM2BLOCKS(Constraint<T>::m_dimension), THREADS_PER_BLOCK>>>(Constraint<T>::m_dimension,
                                                                                          d_vec.raw(),
                                                                                          m_d_lowerBound->raw(),
                                                                                          m_d_upperBound->raw());
    }

    void print() {
        std::cout << "Node: " << Constraint<T>::m_nodeIndex << ", Constraint: Rectangle, \n";
        printIfTensor("Lower bound: ", m_d_lowerBound);
        printIfTensor("Upper bound: ", m_d_upperBound);
    }
};


#endif
