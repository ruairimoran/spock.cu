#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "cones.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t, T *, T *, T *);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectPolyhedron(size_t, T *, T *);


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
    size_t m_matAxis = 2;

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

    virtual void op(DTensor<T> &, DTensor<T> &, DTensor<T> &) {};

    virtual void op(DTensor<T> &, DTensor<T> &) {};

    virtual void adj(DTensor<T> &, DTensor<T> &, DTensor<T> &) {};

    virtual void adj(DTensor<T> &, DTensor<T> &) {};

    friend std::ostream &operator<<(std::ostream &out, const Constraint<T> &data) { return data.print(out); }
};


/**
 * No constraint
 * - used as placeholder
*/
TEMPLATE_WITH_TYPE_T
class NoConstraint : public Constraint<T> {

public:
    explicit NoConstraint() = default;

    void adj(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        k_setToZero<<<numBlocks(x.numEl(), TPB), TPB>>>(x.raw(), x.numEl());
        k_setToZero<<<numBlocks(u.numEl(), TPB), TPB>>>(u.raw(), u.numEl());
    }

    void adj(DTensor<T> &dual, DTensor<T> &x) {
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
    size_t m_numNodesMinus1 = 0;
    size_t m_numStates = 0;
    size_t m_numInputs = 0;

    std::ostream &print(std::ostream &out) const {
        out << "Constraint: Rectangle\n";
        return out;
    }

public:
    explicit Rectangle(std::string path, std::string ext, size_t numNodes, size_t numStates, size_t numInputs) :
        m_numStates(numStates), m_numInputs(numInputs) {
        m_numNodesMinus1 = numNodes - 1;
        m_d_lowerBound = std::make_unique<DTensor<T>>(DTensor<T>::parseFromFile(path + "LB" + ext));
        m_d_upperBound = std::make_unique<DTensor<T>>(DTensor<T>::parseFromFile(path + "UB" + ext));
        this->m_dim = m_d_lowerBound->numEl();
    }

    void constrain(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_projectRectangle<<<numBlocks(this->m_dim, TPB), TPB>>>(this->m_dim, d_vec.raw(),
                                                                 m_d_lowerBound->raw(), m_d_upperBound->raw());
    }

    void op(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        memCpyNode2Node(dual, x, 0, m_numNodesMinus1, m_numStates);
        memCpyNode2Node(dual, u, 0, m_numNodesMinus1, m_numInputs, m_numStates);
    }

    void op(DTensor<T> &dual, DTensor<T> &x) {
        memCpyNode2Node(dual, x, 0, m_numNodesMinus1, m_numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        memCpyNode2Node(x, dual, 0, m_numNodesMinus1, m_numStates);
        memCpyNode2Node(u, dual, 0, m_numNodesMinus1, m_numInputs, 0, m_numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x) {
        memCpyNode2Node(x, dual, 0, m_numNodesMinus1, m_numStates);
    }
};


/**
 * Polyhedron constraint:
 * Gx <= b
 *
 * @param G matrix
 * @param b upper bound
*/
TEMPLATE_WITH_TYPE_T
class Polyhedron : public Constraint<T> {

private:
    std::unique_ptr<DTensor<T>> m_d_gamma = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_upperBound = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuNonleafWorkspace = nullptr;
    size_t m_numNodesMinus1 = 0;
    size_t m_numStates = 0;
    size_t m_numInputs = 0;

    std::ostream &print(std::ostream &out) const {
        out << "Constraint: Polyhedron\n";
        return out;
    }

public:
    explicit Polyhedron(std::string path, std::string ext, ConstraintMode mode, size_t numNodes, size_t numStates,
                        size_t numInputs) : m_numStates(numStates), m_numInputs(numInputs) {
        m_numNodesMinus1 = numNodes - 1;
        /* Read matrix and vector */
        DTensor<T> g(DTensor<T>::parseFromFile(path + "Gamma" + ext));
        DTensor<T> b(DTensor<T>::parseFromFile(path + "UB" + ext));
        /* Create workspace for nonleaf */
        if (mode == nonleaf)
            m_d_xuNonleafWorkspace = std::make_unique<DTensor<T>>(m_numStates + m_numInputs, 1, numNodes);
        /* Create tensor Gamma */
        m_d_gamma = std::make_unique<DTensor<T>>(g.numRows(), g.numCols(), numNodes);
        for (size_t i = 0; i < numNodes; i++) {
            DTensor<T> gNode(*m_d_gamma, this->m_matAxis, i, i);
            g.deviceCopyTo(gNode);
        }
        /* Compute transpose */
        m_d_gammaTr = std::make_unique<DTensor<T>>(g.numCols(), g.numRows(), numNodes);
        DTensor<T> gTr = m_d_gamma->tr();
        gTr.deviceCopyTo(*m_d_gammaTr);
        /* Create upper bound tensor */
        m_d_upperBound = std::make_unique<DTensor<T>>(b.numRows(), 1, numNodes);
        for (size_t i = 0; i < numNodes; i++) {
            DTensor<T> bNode(*m_d_upperBound, this->m_matAxis, i, i);
            b.deviceCopyTo(bNode);
        }
        /* Set constraint dimension */
        this->m_dim = m_d_upperBound->numEl();
    }

    void constrain(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_projectPolyhedron<<<numBlocks(this->m_dim, TPB), TPB>>>(this->m_dim, d_vec.raw(), m_d_upperBound->raw());
    }

    void op(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        memCpyNode2Node(*m_d_xuNonleafWorkspace, x, 0, m_numNodesMinus1, m_numStates);
        memCpyNode2Node(*m_d_xuNonleafWorkspace, u, 0, m_numNodesMinus1, m_numInputs, 0, m_numStates);
        dual.addAB(*m_d_gamma, *m_d_xuNonleafWorkspace);
    }

    void op(DTensor<T> &dual, DTensor<T> &x) {
        dual.addAB(*m_d_gamma, x);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        m_d_xuNonleafWorkspace->addAB(*m_d_gammaTr, dual);
        memCpyNode2Node(x, *m_d_xuNonleafWorkspace, 0, m_numNodesMinus1, m_numStates);
        memCpyNode2Node(u, *m_d_xuNonleafWorkspace, 0, m_numNodesMinus1, m_numInputs, 0, m_numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x) {
        x.addAB(*m_d_gammaTr, dual);
    }
};


#endif
