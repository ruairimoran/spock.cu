#ifndef COSTS_CUH
#define COSTS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "projections.cuh"


TEMPLATE_WITH_TYPE_T
__global__ void k_projectRectangle(size_t, T *, T *, T *);


/**
 * Base class for costs
 */
TEMPLATE_WITH_TYPE_T
class Cost {

protected:
    ScenarioTree<T> &m_tree;
    size_t m_dimPerNode = 0;
    size_t m_numNodes = 0;
    size_t m_dim = 0;
    size_t m_rowAxis = 0;
    size_t m_matAxis = 2;
    std::string m_prefix = "Cost_";
    std::unique_ptr<DTensor<T>> m_d_reshapedData = nullptr;

    explicit Cost(ScenarioTree<T> &tree): m_tree(tree) {};

    virtual std::ostream &print(std::ostream &) const = 0;

public:
    virtual ~Cost() = default;

    size_t dimPerNode() { return m_dimPerNode; }

    size_t dim() { return m_dim; }

    size_t numNodes() { return m_numNodes; }

    virtual bool isQuadratic() { return false; }

    virtual bool isLinear() { return false; }

    virtual void reshape(DTensor<T> &, size_t, size_t) = 0;

    virtual void op(DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &) = 0;

    virtual void op(DTensor<T> &, DTensor<T> &, DTensor<T> &) = 0;

    virtual void adj(DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &) = 0;

    virtual void adj(DTensor<T> &, DTensor<T> &, DTensor<T> &) = 0;

    virtual void translate(DTensor<T> &) = 0;

    virtual void project(DTensor<T> &) = 0;

    friend std::ostream &operator<<(std::ostream &out, const Cost<T> &data) { return data.print(out); }
};


/**
 * Quadratic costs
 */
template<typename T>
class CostQuadratic : public Cost<T> {
private:
    std::unique_ptr<DTensor<T>> m_d_sqrtQ = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtR = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtQLeaf = nullptr;
    std::unique_ptr<DTensor<T>> m_d_translation = nullptr;
    std::unique_ptr<SocProjection<T>> m_socs = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_scalarWorkspace = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Cost: quadratic\n";
        return out;
    }

public:
    explicit CostQuadratic(ScenarioTree<T> &tree, TreePart part = nonleaf): Cost<T>(tree) {
        /* Read data from files */
        this->m_prefix = tree.strOfPart(part) + this->m_prefix;
        DTensor<T> sqrtQ = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtQ" + tree.fpFileExt());
        DTensor<T> trans = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "translation" + tree.fpFileExt());
        m_d_scalarWorkspace = std::make_unique<DTensor<T>>(1, 1, tree.numNodes(), true);
        if (part == nonleaf) {
            this->m_dimPerNode = tree.numStatesAndInputs() + 2;
            this->m_numNodes = tree.numNodes();
            m_d_sqrtQ = std::make_unique<DTensor<T>>(sqrtQ);
            m_d_translation = std::make_unique<DTensor<T>>(trans);
            m_d_sqrtR = std::make_unique<DTensor<T>>(
                DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtR" + tree.fpFileExt()));
            m_d_uWorkspace = std::make_unique<DTensor<T>>(tree.numInputs(), 1, this->m_numNodes, true);
        } else {
            this->m_dimPerNode = tree.numStates() + 2;
            this->m_numNodes = tree.numLeafNodes();
            m_d_sqrtQ = std::make_unique<DTensor<T>>(sqrtQ.numRows(), sqrtQ.numCols(), this->m_numNodes);
            m_d_translation = std::make_unique<DTensor<T>>(trans.numRows(), trans.numCols(), this->m_numNodes);
            for (size_t i = 0; i < this->m_numNodes; i++) {
                DTensor<T> sqrtQ_slice(*m_d_sqrtQ, this->m_matAxis, i, i);
                sqrtQ.deviceCopyTo(sqrtQ_slice);
                DTensor<T> trans_slice(*m_d_translation, this->m_matAxis, i, i);
                trans.deviceCopyTo(trans_slice);
            }
        }
        m_d_xWorkspace = std::make_unique<DTensor<T>>(tree.numStates(), 1, this->m_numNodes, true);
        this->m_dim = this->m_dimPerNode * this->m_numNodes;
    }

    bool isQuadratic() { return true; }

    void reshape(DTensor<T> &work, size_t start, size_t end) {
        /*
         * SocProjection requires one matrix, where the columns are the vectors.
         */
        this->m_d_reshapedData = std::make_unique<DTensor<T>>(work, this->m_rowAxis, start, end);
        this->m_d_reshapedData->reshape(this->m_dimPerNode, this->m_numNodes, 1);
        m_socs = std::make_unique<SocProjection<T>>(*(this->m_d_reshapedData));
    }

    void op(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        /* IV:1 */
        this->m_tree.memCpyAnc2Node(*m_d_xWorkspace, x, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates(), 0, 0);
        m_d_xWorkspace->addAB(*m_d_sqrtQ, *m_d_xWorkspace);
        /* IV:2 */
        this->m_tree.memCpyAnc2Node(*m_d_uWorkspace, u, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(), 0, 0);
        m_d_uWorkspace->addAB(*m_d_sqrtR, *m_d_uWorkspace);
        /* IV:3,4 */
        t *= 0.5;  // This affects the current 't'!!! But it shouldn't matter...
        /* IV (organise IV:1-4) */
        /* :1 */
        memCpyNode2Node(iv, *m_d_xWorkspace, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates());
        /* :2 */
        memCpyNode2Node(iv, *m_d_uWorkspace, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(), this->m_tree.numStates());
        /* :3 */
        memCpyNode2Node(iv, t, 1, this->m_tree.numNodesMinus1(), 1, this->m_tree.numStatesAndInputs());
        /* :4 */
        memCpyNode2Node(iv, t, 1, this->m_tree.numNodesMinus1(), 1, this->m_tree.numStatesAndInputs() + 1);
    }

    void op(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &s) {
        /* VI:1 */
        m_d_xWorkspace->addAB(*m_d_sqrtQ, xLeaf);
        /* VI:2,3 */
        s *= 0.5;  // This affects the current 's'!!! But it shouldn't matter...
        /* VI (organise VI:1-3) */
        /* :1 */
        memCpyNode2Node(vi, *m_d_xWorkspace, 0, this->m_tree.numLeafNodesMinus1(), this->m_tree.numStates());
        /* :2 */
        this->m_tree.memCpyLeaf2Zero(vi, s, 1, this->m_tree.numStates(), 0);
        /* :3 */
        this->m_tree.memCpyLeaf2Zero(vi, s, 1, this->m_tree.numStates() + 1, 0);
    }

    void adj(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        /* -> Compute `Qiv1` at every nonroot node */
        memCpyNode2Node(*m_d_xWorkspace, iv, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates());
        m_d_xWorkspace->addAB(*m_d_sqrtQ, *m_d_xWorkspace);
        /* -> Compute `Riv2` at every nonroot node */
        memCpyNode2Node(*m_d_uWorkspace, iv, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(), 0, this->m_tree.numStates());
        m_d_uWorkspace->addAB(*m_d_sqrtR, *m_d_uWorkspace);
        /* -> Add children of each nonleaf node */
        for (size_t chIdx = 0; chIdx < this->m_tree.numEvents(); chIdx++) {
            /* -> Add to `x` all children `Qiv1` */
            this->m_tree.memCpyCh2Node(x, *m_d_xWorkspace, 0, this->m_tree.numNonleafNodesMinus1(), chIdx, true);
            /* -> Add to `u` all children `Riv2` */
            this->m_tree.memCpyCh2Node(u, *m_d_uWorkspace, 0, this->m_tree.numNonleafNodesMinus1(), chIdx, true);
        }
        /* t */
        memCpyNode2Node(t, iv, 1, this->m_tree.numNodesMinus1(), 1, 0, this->m_tree.numStatesAndInputs());
        memCpyNode2Node(*m_d_scalarWorkspace, iv, 1, this->m_tree.numNodesMinus1(), 1, 0, this->m_tree.numStatesAndInputs() + 1);
        t += *m_d_scalarWorkspace;
        t *= 0.5;
    }

    void adj(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &s) {
        memCpyNode2Node(*m_d_xWorkspace, vi, 0, this->m_tree.numLeafNodesMinus1(), this->m_tree.numStates());
        xLeaf.addAB(*m_d_sqrtQ, *m_d_xWorkspace, 1., 1.);
        this->m_tree.memCpyZero2Leaf(s, vi, 1, 0, this->m_tree.numStates());
        this->m_tree.memCpyZero2Leaf(*m_d_scalarWorkspace, vi, 1, 0, this->m_tree.numStates() + 1);
        DTensor<T> sLeaf(s, this->m_matAxis, this->m_tree.numNonleafNodes(), this->m_tree.numNodesMinus1());
        DTensor<T> wsLeaf(*m_d_scalarWorkspace, this->m_matAxis, this->m_tree.numNonleafNodes(), this->m_tree.numNodesMinus1());
        sLeaf += wsLeaf;
        sLeaf *= 0.5;
    }

    void translate(DTensor<T> &d_vec) {
        d_vec += *m_d_translation;
    }

    void project(DTensor<T> &) {
        m_socs->project(*(this->m_d_reshapedData));
    }
};


/**
 * Linear costs
 */
template<typename T>
class CostLinear : public Cost<T> {
private:
    std::unique_ptr<DTensor<T>> m_d_gradient = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gradientTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_lowerBound = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uWorkspace = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Cost: linear\n";
        return out;
    }

public:
    explicit CostLinear(ScenarioTree<T> &tree, TreePart part = nonleaf): Cost<T>(tree) {
        /* Read data from files */
        this->m_prefix = tree.strOfPart(part) + this->m_prefix;
        DTensor<T> grad = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "gradient" + tree.fpFileExt());
        this->m_dimPerNode = 2;
        if (part == nonleaf) {
            this->m_numNodes = tree.numNodes();
            m_d_gradient = std::make_unique<DTensor<T>>(grad);
            m_d_xuWorkspace = std::make_unique<DTensor<T>>(tree.numStatesAndInputs(), 1, this->m_numNodes, true);
            m_d_xWorkspace = std::make_unique<DTensor<T>>(tree.numStates(), 1, this->m_numNodes, true);
            m_d_uWorkspace = std::make_unique<DTensor<T>>(tree.numInputs(), 1, this->m_numNodes, true);
        } else {
            this->m_numNodes = tree.numLeafNodes();
            m_d_gradient = std::make_unique<DTensor<T>>(grad.numRows(), grad.numCols(), this->m_numNodes);
            for (size_t i = 0; i < this->m_numNodes; i++) {
                DTensor<T> grad_slice(*m_d_gradient, this->m_matAxis, i, i);
                grad.deviceCopyTo(grad_slice);
            }
        }
        m_d_gradientTr = std::make_unique<DTensor<T>>(m_d_gradient->tr());
        this->m_dim = this->m_dimPerNode * this->m_numNodes;
        m_d_lowerBound = std::make_unique<DTensor<T>>(std::vector<T>(this->m_dim, -INFINITY), this->m_dim);
    }

    bool isLinear() { return true; }

    void reshape(DTensor<T> &work, size_t start, size_t end) {
        this->m_d_reshapedData = std::make_unique<DTensor<T>>(work, this->m_rowAxis, start, end);
    }

    void op(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        this->m_tree.memCpyAnc2Node(*m_d_xuWorkspace, x, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates());
        this->m_tree.memCpyAnc2Node(*m_d_xuWorkspace, u, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(), this->m_tree.numStates());
        DTensor<T> iv_xu(iv, this->m_rowAxis, )
        iv.addAB(*m_d_gradient, *m_d_xuWorkspace);
    }

    void op(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &s) {
        vi.addAB(*m_d_gradient, xLeaf);
    }

    void adj(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        m_d_xuWorkspace->addAB(*m_d_gradientTr, iv);
        memCpyNode2Node(*m_d_xWorkspace, *m_d_xuWorkspace, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates());
        memCpyNode2Node(*m_d_uWorkspace, *m_d_xuWorkspace, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(), this->m_tree.numStates());
        /* -> Add children of each nonleaf node */
        for (size_t chIdx = 0; chIdx < this->m_tree.numEvents(); chIdx++) {
            /* -> Add to `x` all children `Qiv1` */
            this->m_tree.memCpyCh2Node(x, *m_d_xWorkspace, 0, this->m_tree.numNonleafNodesMinus1(), chIdx, true);
            /* -> Add to `u` all children `Riv2` */
            this->m_tree.memCpyCh2Node(u, *m_d_uWorkspace, 0, this->m_tree.numNonleafNodesMinus1(), chIdx, true);
        }
    }

    void adj(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &s) {
        xLeaf.addAB(*m_d_gradientTr, vi, 1., 1.);
    }

    void translate(DTensor<T> &) {}

    void project(DTensor<T> &d_upperBound) {
        k_projectRectangle<<<numBlocks(this->m_dim, TPB), TPB>>>(this->m_dim,
                                                                 this->m_d_reshapedData->raw(),
                                                                 m_d_lowerBound->raw(),
                                                                 d_upperBound.raw());
    }
};


#endif
