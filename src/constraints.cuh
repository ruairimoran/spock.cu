#ifndef CONSTRAINTS_CUH
#define CONSTRAINTS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "cones.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectPolyhedron(size_t, T *, T *, T *);


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
    size_t m_matAxis = 2;
    size_t m_dim = 0;
    size_t m_dimPerNode = 0;
    size_t m_numNodes = 0;
    size_t m_numNodesMinus1 = 0;
    size_t m_numStates = 0;
    size_t m_numInputs = 0;
    std::unique_ptr<DTensor<T>> m_d_lowerBound = nullptr;
    std::unique_ptr<DTensor<T>> m_d_upperBound = nullptr;

    explicit Constraint(size_t numNodes, size_t numStates, size_t numInputs) :
        m_numNodes(numNodes), m_numStates(numStates), m_numInputs(numInputs) {
        m_numNodesMinus1 = m_numNodes - 1;
    }

    void makeBounds(size_t dimPerNode, size_t n) {
        m_dimPerNode = dimPerNode;
        m_d_lowerBound = std::make_unique<DTensor<T>>(m_dimPerNode, 1, n);
        m_d_upperBound = std::make_unique<DTensor<T>>(m_dimPerNode, 1, n);
        m_dim = m_d_lowerBound->numEl();
    }

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

    virtual size_t dimensionPerNode() { return m_dimPerNode; }

    virtual void op(DTensor<T> &, DTensor<T> &, DTensor<T> &) {};

    virtual void op(DTensor<T> &, DTensor<T> &) {};

    virtual void adj(DTensor<T> &, DTensor<T> &, DTensor<T> &) {};

    virtual void adj(DTensor<T> &, DTensor<T> &) {};

    void constrain(DTensor<T> &d_vec) {
        dimensionCheck(d_vec);
        k_projectPolyhedron<<<numBlocks(this->m_dim, TPB), TPB>>>(this->m_dim, d_vec.raw(),
                                                                  m_d_lowerBound->raw(), m_d_upperBound->raw());
    }

    friend std::ostream &operator<<(std::ostream &out, const Constraint<T> &data) { return data.print(out); }
};


/**
 * No constraint
 * - used as placeholder
*/
TEMPLATE_WITH_TYPE_T
class NoConstraint : public Constraint<T> {

public:
    explicit NoConstraint() : Constraint<T>(0, 0, 0) {};

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
    std::ostream &print(std::ostream &out) const {
        out << "Constraint: Rectangle\n";
        return out;
    }

public:
    explicit Rectangle(std::string path, std::string ext,
                       size_t numNodes, size_t numStates, size_t numInputs) :
        Constraint<T>(numNodes, numStates, numInputs) {
        DTensor<T> lb(DTensor<T>::parseFromFile(path + "LB" + ext));
        DTensor<T> ub(DTensor<T>::parseFromFile(path + "UB" + ext));
        /* Create bounds memory */
        this->makeBounds(lb.numEl(), this->m_numNodes);
        /* Fill bounds */
        for (size_t i = 0; i < numNodes; i++) {
            DTensor<T> lbNode(*this->m_d_lowerBound, this->m_matAxis, i, i);
            DTensor<T> ubNode(*this->m_d_lowerBound, this->m_matAxis, i, i);
            lb.deviceCopyTo(lbNode);
            ub.deviceCopyTo(ubNode);
        }
    }

    void op(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        memCpyNode2Node(dual, x, 0, this->m_numNodesMinus1, this->m_numStates);
        memCpyNode2Node(dual, u, 0, this->m_numNodesMinus1, this->m_numInputs, this->m_numStates);
    }

    void op(DTensor<T> &dual, DTensor<T> &x) {
        memCpyNode2Node(dual, x, 0, this->m_numNodesMinus1, this->m_numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        memCpyNode2Node(x, dual, 0, this->m_numNodesMinus1, this->m_numStates);
        memCpyNode2Node(u, dual, 0, this->m_numNodesMinus1, this->m_numInputs, 0, this->m_numStates);
    }

    void adj(DTensor<T> &dual, DTensor<T> &x) {
        memCpyNode2Node(x, dual, 0, this->m_numNodesMinus1, this->m_numStates);
    }
};


/**
 * Polyhedron constraint:
 * lb <= Gz <= ub
 *
 * @param G matrix
 * @param lb lower bound
 * @param ub upper bound
*/
TEMPLATE_WITH_TYPE_T
class Polyhedron : public Constraint<T> {

private:
    std::unique_ptr<DTensor<T>> m_d_gamma = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuNonleafWorkspace = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Constraint: Polyhedron\n";
        return out;
    }

public:
    explicit Polyhedron(std::string path, std::string ext, ConstraintMode mode,
                        size_t numNodes, size_t numStates, size_t numInputs) :
        Constraint<T>(numNodes, numStates, numInputs) {
        /* Read matrix and vector */
        DTensor<T> g(DTensor<T>::parseFromFile(path + "Gamma" + ext));
        DTensor<T> lb(DTensor<T>::parseFromFile(path + "LB" + ext));
        DTensor<T> ub(DTensor<T>::parseFromFile(path + "UB" + ext));
        /* Create workspace for nonleaf */
        if (mode == nonleaf)
            m_d_xuNonleafWorkspace = std::make_unique<DTensor<T>>(this->m_numStates + this->m_numInputs, 1,
                                                                  this->m_numNodes);
        /* Create tensor m_d_gamma = (Γ, Γ, ..., Γ) */
        m_d_gamma = std::make_unique<DTensor<T>>(g.numRows(), g.numCols(), this->m_numNodes);
        for (size_t i = 0; i < this->m_numNodes; i++) {
            DTensor<T> gNode(*m_d_gamma, this->m_matAxis, i, i);
            g.deviceCopyTo(gNode);
        }
        /* Compute transpose m_d_gammaTr = (Γ', Γ', ..., Γ') */
        m_d_gammaTr = std::make_unique<DTensor<T>>(g.numCols(), g.numRows(), this->m_numNodes);
        DTensor<T> gTr = m_d_gamma->tr();
        gTr.deviceCopyTo(*m_d_gammaTr);
        /* Create bound tensors m_d_lowerBound = (lb, lb, ..., lb) and m_d_upperBound = (ub, ub, ..., ub) */
        this->makeBounds(lb.numEl(), this->m_numNodes);
        for (size_t i = 0; i < this->m_numNodes; i++) {
            DTensor<T> lbNode(*this->m_d_lowerBound, this->m_matAxis, i, i);
            DTensor<T> ubNode(*this->m_d_upperBound, this->m_matAxis, i, i);
            lb.deviceCopyTo(lbNode);
            ub.deviceCopyTo(ubNode);
        }
    }

    /**
     * Operator: dual <- [Γxi Γui] * [xi' ui']' for nonleaf nodes
     * @param dual result space
     * @param x nonleaf states
     * @param u inputs
     */
    void op(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        memCpyNode2Node(*m_d_xuNonleafWorkspace, x, 0, this->m_numNodesMinus1, this->m_numStates);
        memCpyNode2Node(*m_d_xuNonleafWorkspace, u, 0, this->m_numNodesMinus1, this->m_numInputs, this->m_numStates);
        dual.addAB(*m_d_gamma, *m_d_xuNonleafWorkspace);
    }

    /**
     * Operator: dual <- Γxj * xj for leaf nodes
     * @param dual result space
     * @param x leaf states
     */
    void op(DTensor<T> &dual, DTensor<T> &x) {
        dual.addAB(*m_d_gamma, x);
    }

    /**
     * Operator: xi, ui <- [Γxi Γui]' * dual for nonleaf nodes
     * @param dual projected vector
     * @param x nonleaf states
     * @param u inputs
     */
    void adj(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        m_d_xuNonleafWorkspace->addAB(*m_d_gammaTr, dual);
        memCpyNode2Node(x, *m_d_xuNonleafWorkspace, 0, this->m_numNodesMinus1, this->m_numStates);
        memCpyNode2Node(u, *m_d_xuNonleafWorkspace, 0, this->m_numNodesMinus1, this->m_numInputs, 0, this->m_numStates);
    }

    /**
     * Operator: xj <- Γxj' * dual for leaf nodes
     * @param dual projected vector
     * @param x leaf states
     */
    void adj(DTensor<T> &dual, DTensor<T> &x) {
        x.addAB(*m_d_gammaTr, dual);
    }
};


/**
 * Polyhedron with identity constraint:
 * lb <= [I G']' * z <= ub
 *
 * @param I identity matrix
 * @param G matrix
 * @param lb lower bound
 * @param ub upper bound
*/
TEMPLATE_WITH_TYPE_T
class PolyhedronWithIdentity : public Constraint<T> {

private:
    std::unique_ptr<DTensor<T>> m_d_gamma = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaTr = nullptr;
    size_t m_numNodesDoubled = 0;
    size_t m_numNodesDoubledMinus1 = 0;

    std::ostream &print(std::ostream &out) const {
        out << "Constraint: Polyhedron with identity\n";
        return out;
    }

public:
    explicit PolyhedronWithIdentity(std::string path, std::string ext,
                                    size_t numNodes, size_t numStates, size_t numInputs) :
        Constraint<T>(numNodes, numStates, numInputs) {
        m_numNodesDoubled = this->m_numNodes * 2;
        m_numNodesDoubledMinus1 = m_numNodesDoubled - 1;
        /* Read matrix and vectors */
        DTensor<T> ilb(DTensor<T>::parseFromFile(path + "ILB" + ext));
        DTensor<T> iub(DTensor<T>::parseFromFile(path + "IUB" + ext));
        DTensor<T> g(DTensor<T>::parseFromFile(path + "Gamma" + ext));
        DTensor<T> glb(DTensor<T>::parseFromFile(path + "GLB" + ext));
        DTensor<T> gub(DTensor<T>::parseFromFile(path + "GUB" + ext));
        /* Create tensor m_d_gamma = (Γ, Γ, ..., Γ) */
        m_d_gamma = std::make_unique<DTensor<T>>(g.numRows(), g.numCols(), this->m_numNodes);
        for (size_t i = 0; i < this->m_numNodes; i++) {
            DTensor<T> gNode(*m_d_gamma, this->m_matAxis, i, i);
            g.deviceCopyTo(gNode);
        }
        /* Compute transpose m_d_gammaTr = (Γ', Γ', ..., Γ') */
        m_d_gammaTr = std::make_unique<DTensor<T>>(g.numCols(), g.numRows(), this->m_numNodes);
        DTensor<T> gTr = m_d_gamma->tr();
        gTr.deviceCopyTo(*m_d_gammaTr);
        /* Create bound tensors m_d_lowerBound = (ilb, ..., ilb, glb, ..., glb)
         * and m_d_upperBound = (iub, ..., iub, gub, ..., gub) */
        this->makeBounds(ilb.numEl() + glb.numEl(), m_numNodesDoubled);
        for (size_t i = 0; i < this->m_numNodes; i++) {
            DTensor<T> ilbNode(*this->m_d_lowerBound, this->m_matAxis, i, i);
            DTensor<T> iubNode(*this->m_d_upperBound, this->m_matAxis, i, i);
            ilb.deviceCopyTo(ilbNode);
            iub.deviceCopyTo(iubNode);
            size_t idx = i + this->m_numNodes;
            DTensor<T> glbNode(*this->m_d_lowerBound, this->m_matAxis, idx, idx);
            DTensor<T> gubNode(*this->m_d_upperBound, this->m_matAxis, idx, idx);
            glb.deviceCopyTo(glbNode);
            gub.deviceCopyTo(gubNode);
        }
    }

    /**
     * Operator: dual <- [I (Γxi Γui)']' * [xi' ui']' for nonleaf nodes
     * @param dual result space
     * @param x nonleaf states
     * @param u inputs
     */
    void op(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        DTensor<T> iDual(dual, this->m_matAxis, 0, this->m_numNodesMinus1);
        DTensor<T> gDual(dual, this->m_matAxis, this->m_numNodes, m_numNodesDoubledMinus1);
        memCpyNode2Node(iDual, x, 0, this->m_numNodesMinus1, this->m_numStates);
        memCpyNode2Node(iDual, u, 0, this->m_numNodesMinus1, this->m_numInputs, this->m_numStates);
        gDual.addAB(*m_d_gamma, iDual);
    }

    /**
     * Operator: dual <- Γxj * xj for leaf nodes
     * @param dual result space
     * @param x leaf states
     */
    void op(DTensor<T> &dual, DTensor<T> &x) {
        DTensor<T> iDual(dual, this->m_matAxis, 0, this->m_numNodesMinus1);
        DTensor<T> gDual(dual, this->m_matAxis, this->m_numNodes, m_numNodesDoubledMinus1);
        memCpyNode2Node(iDual, x, 0, this->m_numNodesMinus1, this->m_numStates);
        gDual.addAB(*m_d_gamma, x);
    }

    /**
     * Operator: xi, ui <- [I (Γxi Γui)']' * dual for nonleaf nodes
     * @param dual projected vector
     * @param x nonleaf states
     * @param u inputs
     */
    void adj(DTensor<T> &dual, DTensor<T> &x, DTensor<T> &u) {
        DTensor<T> iDual(dual, this->m_matAxis, 0, this->m_numNodesMinus1);
        DTensor<T> gDual(dual, this->m_matAxis, this->m_numNodes, m_numNodesDoubledMinus1);
        iDual.addAB(*m_d_gammaTr, gDual, 1., 1.);
        memCpyNode2Node(x, iDual, 0, this->m_numNodesMinus1, this->m_numStates);
        memCpyNode2Node(u, iDual, 0, this->m_numNodesMinus1, this->m_numInputs, 0, this->m_numStates);
    }

    /**
     * Operator: xj <- [I (Γxj)']' * dual for leaf nodes
     * @param dual projected vector
     * @param x leaf states
     */
    void adj(DTensor<T> &dual, DTensor<T> &x) {
        DTensor<T> iDual(dual, this->m_matAxis, 0, this->m_numNodesMinus1);
        DTensor<T> gDual(dual, this->m_matAxis, this->m_numNodes, m_numNodesDoubledMinus1);
        iDual.addAB(*m_d_gammaTr, gDual, 1., 1.);
        memCpyNode2Node(x, iDual, 0, this->m_numNodesMinus1, this->m_numStates);
    }
};


#endif
