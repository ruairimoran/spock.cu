#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t, T *, T *, T *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyNode2Node(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyAnc2Node(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyLeaf2ZeroLeaf(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyZeroLeaf2Leaf(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyCh2Node(T *, T *, size_t, size_t, size_t, size_t, size_t *, size_t *, bool);

/**
 * Memory copy for trees
 */
enum MemCpyMode {
    node2Node,  ///< transfer node data to same node index
    anc2Node,  ///< transfer ancestor data to node index
    leaf2ZeroLeaf,  ///< transfer leaf data to zero-indexed leaf nodes
    zeroLeaf2Leaf,  ///< transfer zero-indexed leaf data to leaf nodes
    defaultMode = node2Node
};

TEMPLATE_WITH_TYPE_T
void memCpy(DTensor<T> *dst, DTensor<T> *src,
            size_t nodeFrom, size_t nodeTo, size_t numEl,
            size_t elFromDst = 0, size_t elFromSrc = 0,
            MemCpyMode mode = MemCpyMode::defaultMode,
            DTensor<size_t> *ancestors = nullptr,
            DTensor<size_t> *chFrom = nullptr, DTensor<size_t> *chTo = nullptr) {
    size_t nodeSizeDst = dst->numRows();
    size_t nodeSizeSrc = src->numRows();
    if (dst->numCols() != 1 || src->numCols() != 1) throw std::invalid_argument("[memCpy] numCols must be 1.");
    if (std::max(nodeSizeDst, nodeSizeSrc) > TPB) throw std::invalid_argument("[memCpy] Node data too large.");
    if (mode == anc2Node && nodeFrom < 1) throw std::invalid_argument("[memCpy] Root node has no ancestor.");
    size_t nBlocks = nodeTo + 1;
    if (mode == node2Node) {
        k_memCpyNode2Node<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst, nodeSizeSrc,
                                            elFromDst, elFromSrc);
    }
    if (mode == anc2Node) {
        k_memCpyAnc2Node<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst, nodeSizeSrc,
                                           elFromDst, elFromSrc, ancestors->raw());
    }
    /**
     * For leaf transfers, you must transfer all leaf nodes! So `nodeFrom` == numNonleafNodes.
     * The `nodeFrom/To` requires the actual node numbers (not zero-indexed).
     */
    if (mode == leaf2ZeroLeaf) {
        k_memCpyLeaf2ZeroLeaf<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst,
                                                nodeSizeSrc, elFromDst, elFromSrc);
    }
    if (mode == zeroLeaf2Leaf) {
        k_memCpyZeroLeaf2Leaf<<<nBlocks, TPB>>>(dst->raw(), src->raw(), nodeFrom, nodeTo, numEl, nodeSizeDst,
                                                nodeSizeSrc, elFromDst, elFromSrc);
    }
}


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
    explicit Rectangle(std::string file) {
        m_d_lowerBound = std::make_unique<DTensor<T>>(DTensor<T>::parseFromTextFile(file + "LB", rowMajor));
        m_d_upperBound = std::make_unique<DTensor<T>>(DTensor<T>::parseFromTextFile(file + "UB", rowMajor));
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
