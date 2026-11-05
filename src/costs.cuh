#ifndef COSTS_CUH
#define COSTS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "projections.cuh"
#include "cost_kernels.cuh"
#include <stdexcept>


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

    explicit Cost(ScenarioTree<T> &tree) : m_tree(tree) {};

    static void transformWorkspace(const DTensor<T> &matrix,
                                   std::unique_ptr<DTensor<T>> &workspace,
                                   std::unique_ptr<DTensor<T>> &product) {
        // GEMM output must not alias its input. Swap owning buffers instead
        // of allocating or copying a temporary on each operator evaluation.
        product->addAB(matrix, *workspace);
        workspace.swap(product);
    }

    static bool diagonalFactor(ScenarioTree<T> &tree, TreePart part, const char *name) {
        std::ifstream file(tree.path() + tree.json());
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        rapidjson::Document metadata;
        metadata.Parse(text.c_str());
        if (metadata.HasParseError() || !metadata.HasMember("cost"))
            throw std::invalid_argument("Invalid cost metadata");
        const std::string key = tree.strOfPart(part) + name;
        const auto &cost = metadata["cost"];
        if (!cost.HasMember(key.c_str())) return false;  // Legacy files contain dense factors.
        const auto &value = cost[key.c_str()];
        if (!value.IsString()) throw std::invalid_argument("Invalid cost factor structure");
        const std::string structure = value.GetString();
        if (structure != "dense" && structure != "diagonal")
            throw std::invalid_argument("Unknown cost factor structure");
        return structure == "diagonal";
    }

    static void validateFactor(const DTensor<T> &factor, size_t size, size_t nodes, bool diagonal) {
        if (factor.numRows() != size || factor.numCols() != (diagonal ? 1 : size) || factor.numMats() != nodes)
            throw std::invalid_argument("Cost factor dimensions do not match its structure or tree");
    }

    void forwardFactor(DTensor<T> &output, DTensor<T> &input, const DTensor<T> &factor,
                       std::unique_ptr<DTensor<T>> &workspace, std::unique_ptr<DTensor<T>> &product,
                       size_t size, size_t offset, bool diagonal, bool leaf) {
        if (diagonal) {
            const size_t count = size * (m_numNodes - (leaf ? 0 : 1));
            cost_detail::diagonalForward<<<numBlocks(count, TPB), TPB>>>(output.raw(), input.raw(), factor.raw(),
                m_tree.d_ancestors().raw(), size, m_numNodes, m_dimPerNode, offset, leaf);
            gpuErrChk(cudaGetLastError());
        } else {
            if (leaf) workspace->addAB(factor, input);
            else {
                m_tree.memCpyAnc2Node(*workspace, input, 1, m_tree.numNodesMinus1(), size, 0, 0);
                transformWorkspace(factor, workspace, product);
            }
            memCpyNode2Node(output, *workspace, leaf ? 0 : 1, m_numNodes - 1, size, offset);
        }
    }

    void adjointFactor(DTensor<T> &output, DTensor<T> &dual, const DTensor<T> &factor,
                       std::unique_ptr<DTensor<T>> &workspace, std::unique_ptr<DTensor<T>> &product,
                       size_t size, size_t offset, bool diagonal, bool leaf, const T *linear = nullptr) {
        const size_t scalarOffset = leaf ? m_tree.numStates() : m_tree.numStatesAndInputs();
        if (diagonal) {
            const size_t destinations = leaf ? m_numNodes : m_tree.numNonleafNodes();
            cost_detail::diagonalAdjoint<<<numBlocks(size * destinations, TPB), TPB>>>(output.raw(), dual.raw(),
                factor.raw(), linear, m_tree.d_childFrom().raw(), m_tree.d_numChildren().raw(),
                size, destinations, m_dimPerNode, offset, scalarOffset, leaf);
            gpuErrChk(cudaGetLastError());
        } else {
            memCpyNode2Node(*workspace, dual, leaf ? 0 : 1, m_numNodes - 1, size, 0, offset);
            if (leaf) {
                output.addAB(factor, *workspace, T(1), T(1));
                if (linear) {
                    cost_detail::linearAdjoint<<<numBlocks(size * m_numNodes, TPB), TPB>>>(output.raw(), dual.raw(),
                        linear, size, m_numNodes, m_dimPerNode, scalarOffset, true);
                    gpuErrChk(cudaGetLastError());
                }
            } else {
                transformWorkspace(factor, workspace, product);
                if (linear) {
                    cost_detail::linearAdjoint<<<numBlocks(size * m_numNodes, TPB), TPB>>>(workspace->raw(), dual.raw(),
                        linear, size, m_numNodes, m_dimPerNode, scalarOffset, false);
                    gpuErrChk(cudaGetLastError());
                }
                for (size_t child = 0; child < m_tree.numEvents(); ++child)
                    m_tree.memCpyCh2Node(output, *workspace, 0, m_tree.numNonleafNodesMinus1(), child, true);
            }
        }
    }

    void forwardScalars(DTensor<T> &output, DTensor<T> &scalar, DTensor<T> &x, DTensor<T> *u,
                        bool leaf, const T *q = nullptr, const T *r = nullptr) {
        cost_detail::forwardScalars<<<numBlocks(m_numNodes, TPB), TPB>>>(output.raw(), scalar.raw(), x.raw(),
            u ? u->raw() : nullptr, q, r, m_tree.d_ancestors().raw(), m_tree.numStates(), m_tree.numInputs(),
            m_numNodes, m_dimPerNode, leaf);
        gpuErrChk(cudaGetLastError());
    }

    void adjointScalars(DTensor<T> &output, DTensor<T> &dual, bool leaf) {
        cost_detail::adjointScalars<<<numBlocks(m_numNodes, TPB), TPB>>>(output.raw(), dual.raw(), m_numNodes,
            m_dimPerNode, leaf ? m_tree.numStates() : m_tree.numStatesAndInputs(), leaf);
        gpuErrChk(cudaGetLastError());
    }

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

    virtual void project() = 0;

    friend std::ostream &operator<<(std::ostream &out, const Cost<T> &data) { return data.print(out); }
};


/**
 * Quadratic costs
 */
template<typename T>
class CostQuadratic : public Cost<T> {
private:
    bool m_qDiagonal = false, m_rDiagonal = false;
    std::unique_ptr<DTensor<T>> m_d_sqrtQ = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtR = nullptr;
    std::unique_ptr<DTensor<T>> m_d_translation = nullptr;
    std::unique_ptr<SocProjection<T>> m_socs = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xProduct = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uProduct = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Cost: quadratic\n";
        return out;
    }

public:
    explicit CostQuadratic(ScenarioTree<T> &tree, TreePart part = nonleaf) : Cost<T>(tree) {
        /* Read data from files */
        this->m_prefix = tree.strOfPart(part) + this->m_prefix;
        m_qDiagonal = this->diagonalFactor(tree, part, "Q");
        if (part == nonleaf) m_rDiagonal = this->diagonalFactor(tree, part, "R");
        DTensor<T> sqrtQ = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtQ" + tree.fpFileExt());
        this->validateFactor(sqrtQ, tree.numStates(), part == nonleaf ? tree.numNodes() : 1, m_qDiagonal);
        DTensor<T> trans = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "translation" + tree.fpFileExt());
        if (part == nonleaf) {
            this->m_dimPerNode = tree.numStatesAndInputs() + 2;
            this->m_numNodes = tree.numNodes();
            m_d_sqrtQ = std::make_unique<DTensor<T>>(sqrtQ);
            m_d_translation = std::make_unique<DTensor<T>>(trans);
            m_d_sqrtR = std::make_unique<DTensor<T>>(
                DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtR" + tree.fpFileExt()));
            this->validateFactor(*m_d_sqrtR, tree.numInputs(), tree.numNodes(), m_rDiagonal);
            if (!m_rDiagonal) m_d_uWorkspace = std::make_unique<DTensor<T>>(tree.numInputs(), 1, this->m_numNodes, true);
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
        if (!m_qDiagonal) m_d_xWorkspace = std::make_unique<DTensor<T>>(tree.numStates(), 1, this->m_numNodes, true);
        if (part == nonleaf) {
            if (!m_qDiagonal) m_d_xProduct = std::make_unique<DTensor<T>>(tree.numStates(), 1, this->m_numNodes);
            if (!m_rDiagonal) m_d_uProduct = std::make_unique<DTensor<T>>(tree.numInputs(), 1, this->m_numNodes);
        }
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
        this->forwardFactor(iv, x, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, false);
        this->forwardFactor(iv, u, *m_d_sqrtR, m_d_uWorkspace, m_d_uProduct,
                            this->m_tree.numInputs(), this->m_tree.numStates(), m_rDiagonal, false);
        this->forwardScalars(iv, t, x, &u, false);
    }

    void op(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &sLeaf) {
        this->forwardFactor(vi, xLeaf, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, true);
        this->forwardScalars(vi, sLeaf, xLeaf, nullptr, true);
    }

    void adj(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        this->adjointFactor(x, iv, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, false);
        this->adjointFactor(u, iv, *m_d_sqrtR, m_d_uWorkspace, m_d_uProduct,
                            this->m_tree.numInputs(), this->m_tree.numStates(), m_rDiagonal, false);
        this->adjointScalars(t, iv, false);
    }

    void adj(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &sLeaf) {
        this->adjointFactor(xLeaf, vi, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, true);
        this->adjointScalars(sLeaf, vi, true);
    }

    void translate(DTensor<T> &d_vec) {
        d_vec += *m_d_translation;
    }

    void project() {
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
    std::unique_ptr<DTensor<T>> m_d_sum = nullptr;
    std::unique_ptr<DTensor<T>> m_d_set = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xuWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uWorkspace = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Cost: linear\n";
        return out;
    }

public:
    explicit CostLinear(ScenarioTree<T> &tree, TreePart part = nonleaf) : Cost<T>(tree) {
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
        m_d_lowerBound = std::make_unique<DTensor<T>>(std::vector<T>(this->m_numNodes, -INFINITY), this->m_numNodes);
    }

    bool isLinear() { return true; }

    void reshape(DTensor<T> &work, size_t start, size_t end) {
        m_d_sum = std::make_unique<DTensor<T>>(work, this->m_rowAxis, start, start + this->m_numNodes - 1);
        m_d_sum->reshape(1, 1, this->m_numNodes);
        m_d_set = std::make_unique<DTensor<T>>(work, this->m_rowAxis, start + this->m_numNodes, end);
        m_d_set->reshape(1, 1, this->m_numNodes);
    }

    void op(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        this->m_tree.memCpyAnc2Node(*m_d_xuWorkspace, x, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates());
        this->m_tree.memCpyAnc2Node(*m_d_xuWorkspace, u, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(),
                                    this->m_tree.numStates());
        m_d_sum->addAB(*m_d_gradient, *m_d_xuWorkspace);
        t.deviceCopyTo(*m_d_set);
    }

    void op(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &sLeaf) {
        m_d_sum->addAB(*m_d_gradient, xLeaf);
        sLeaf.deviceCopyTo(*m_d_set);
    }

    void adj(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        m_d_xuWorkspace->addAB(*m_d_gradientTr, *m_d_sum);
        memCpyNode2Node(*m_d_xWorkspace, *m_d_xuWorkspace, 1, this->m_tree.numNodesMinus1(), this->m_tree.numStates());
        memCpyNode2Node(*m_d_uWorkspace, *m_d_xuWorkspace, 1, this->m_tree.numNodesMinus1(), this->m_tree.numInputs(),
                        0, this->m_tree.numStates());
        /* -> Add children of each nonleaf node */
        for (size_t chIdx = 0; chIdx < this->m_tree.numEvents(); chIdx++) {
            /* -> Add to `x` all children `Qiv1` */
            this->m_tree.memCpyCh2Node(x, *m_d_xWorkspace, 0, this->m_tree.numNonleafNodesMinus1(), chIdx, true);
            /* -> Add to `u` all children `Riv2` */
            this->m_tree.memCpyCh2Node(u, *m_d_uWorkspace, 0, this->m_tree.numNonleafNodesMinus1(), chIdx, true);
        }
        m_d_set->deviceCopyTo(t);
    }

    void adj(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &sLeaf) {
        xLeaf.addAB(*m_d_gradientTr, *m_d_sum, 1., 1.);
        m_d_set->deviceCopyTo(sLeaf);
    }

    void translate(DTensor<T> &) {}

    void project() {
        k_projectRectangle<<<numBlocks(this->m_numNodes, TPB), TPB>>>(this->m_numNodes,
                                                                      this->m_d_sum->raw(),
                                                                      m_d_lowerBound->raw(),
                                                                      m_d_set->raw());
    }
};


/**
 * Quadratic-plus-linear costs
 */
template<typename T>
class CostQuadraticPlusLinear : public Cost<T> {
private:
    bool m_qDiagonal = false, m_rDiagonal = false;
    std::unique_ptr<DTensor<T>> m_d_sqrtQ = nullptr;
    std::unique_ptr<DTensor<T>> m_d_q = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtR = nullptr;
    std::unique_ptr<DTensor<T>> m_d_r = nullptr;
    std::unique_ptr<DTensor<T>> m_d_translation = nullptr;
    std::unique_ptr<SocProjection<T>> m_socs = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xWorkspace = nullptr;
    std::unique_ptr<DTensor<T>> m_d_xProduct = nullptr;
    std::unique_ptr<DTensor<T>> m_d_uProduct = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Cost: quadratic-plus-linear\n";
        return out;
    }

public:
    explicit CostQuadraticPlusLinear(ScenarioTree<T> &tree, TreePart part = nonleaf) : Cost<T>(tree) {
        /* Read data from files */
        this->m_prefix = tree.strOfPart(part) + this->m_prefix;
        m_qDiagonal = this->diagonalFactor(tree, part, "Q");
        if (part == nonleaf) m_rDiagonal = this->diagonalFactor(tree, part, "R");
        DTensor<T> sqrtQ = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtQ" + tree.fpFileExt());
        DTensor<T> q = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "q" + tree.fpFileExt());
        this->validateFactor(sqrtQ, tree.numStates(), part == nonleaf ? tree.numNodes() : 1, m_qDiagonal);
        DTensor<T> trans = DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "translation" + tree.fpFileExt());
        if (part == nonleaf) {
            this->m_dimPerNode = tree.numStatesAndInputs() + 2;
            this->m_numNodes = tree.numNodes();
            m_d_sqrtQ = std::make_unique<DTensor<T>>(sqrtQ);
            m_d_q = std::make_unique<DTensor<T>>(q);
            m_d_translation = std::make_unique<DTensor<T>>(trans);
            m_d_sqrtR = std::make_unique<DTensor<T>>(
                DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtR" + tree.fpFileExt()));
            this->validateFactor(*m_d_sqrtR, tree.numInputs(), tree.numNodes(), m_rDiagonal);
            m_d_r = std::make_unique<DTensor<T>>(
                DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "r" + tree.fpFileExt()));
            if (!m_rDiagonal) m_d_uWorkspace = std::make_unique<DTensor<T>>(tree.numInputs(), 1, this->m_numNodes, true);
        } else {
            this->m_dimPerNode = tree.numStates() + 2;
            this->m_numNodes = tree.numLeafNodes();
            m_d_sqrtQ = std::make_unique<DTensor<T>>(sqrtQ.numRows(), sqrtQ.numCols(), this->m_numNodes);
            m_d_q = std::make_unique<DTensor<T>>(q.numRows(), q.numCols(), this->m_numNodes);
            m_d_translation = std::make_unique<DTensor<T>>(trans.numRows(), trans.numCols(), this->m_numNodes);
            for (size_t i = 0; i < this->m_numNodes; i++) {
                DTensor<T> sqrtQ_slice(*m_d_sqrtQ, this->m_matAxis, i, i);
                sqrtQ.deviceCopyTo(sqrtQ_slice);
                DTensor<T> q_slice(*m_d_q, this->m_matAxis, i, i);
                q.deviceCopyTo(q_slice);
                DTensor<T> trans_slice(*m_d_translation, this->m_matAxis, i, i);
                trans.deviceCopyTo(trans_slice);
            }
        }
        if (!m_qDiagonal) m_d_xWorkspace = std::make_unique<DTensor<T>>(tree.numStates(), 1, this->m_numNodes, true);
        if (part == nonleaf) {
            if (!m_qDiagonal) m_d_xProduct = std::make_unique<DTensor<T>>(tree.numStates(), 1, this->m_numNodes);
            if (!m_rDiagonal) m_d_uProduct = std::make_unique<DTensor<T>>(tree.numInputs(), 1, this->m_numNodes);
        }
        this->m_dim = this->m_dimPerNode * this->m_numNodes;
    }

    bool isQuadraticPlusLinear() { return true; }

    void reshape(DTensor<T> &work, size_t start, size_t end) {
        /*
         * SocProjection requires one matrix, where the columns are the vectors.
         */
        this->m_d_reshapedData = std::make_unique<DTensor<T>>(work, this->m_rowAxis, start, end);
        this->m_d_reshapedData->reshape(this->m_dimPerNode, this->m_numNodes, 1);
        m_socs = std::make_unique<SocProjection<T>>(*(this->m_d_reshapedData));
    }

    void op(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        this->forwardFactor(iv, x, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, false);
        this->forwardFactor(iv, u, *m_d_sqrtR, m_d_uWorkspace, m_d_uProduct,
                            this->m_tree.numInputs(), this->m_tree.numStates(), m_rDiagonal, false);
        this->forwardScalars(iv, t, x, &u, false, m_d_q->raw(), m_d_r->raw());
    }

    void op(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &sLeaf) {
        this->forwardFactor(vi, xLeaf, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, true);
        this->forwardScalars(vi, sLeaf, xLeaf, nullptr, true, m_d_q->raw());
    }

    void adj(DTensor<T> &iv, DTensor<T> &x, DTensor<T> &u, DTensor<T> &t) {
        this->adjointFactor(x, iv, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, false, m_d_q->raw());
        this->adjointFactor(u, iv, *m_d_sqrtR, m_d_uWorkspace, m_d_uProduct,
                            this->m_tree.numInputs(), this->m_tree.numStates(), m_rDiagonal, false, m_d_r->raw());
        this->adjointScalars(t, iv, false);
    }

    void adj(DTensor<T> &vi, DTensor<T> &xLeaf, DTensor<T> &sLeaf) {
        this->adjointFactor(xLeaf, vi, *m_d_sqrtQ, m_d_xWorkspace, m_d_xProduct,
                            this->m_tree.numStates(), 0, m_qDiagonal, true, m_d_q->raw());
        this->adjointScalars(sLeaf, vi, true);
    }

    void translate(DTensor<T> &d_vec) {
        d_vec += *m_d_translation;
    }

    void project() {
        m_socs->project(*(this->m_d_reshapedData));
    }
};


#endif
