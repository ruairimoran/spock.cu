#ifndef COSTS_CUH
#define COSTS_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "projections.cuh"


/**
 * Base class for costs
 */
template<typename T>
class Cost {

protected:
    size_t m_dimPerNode = 0;
    size_t m_numNodes = 0;
    size_t m_dim = 0;
    std::string m_prefix = "Cost_";
    std::unique_ptr<DTensor<T>> m_d_sqrtQ = nullptr;
    std::unique_ptr<DTensor<T>> m_d_translation = nullptr;
    std::unique_ptr<SocProjection<T>> m_socs = nullptr;

    explicit Cost(ScenarioTree<T> &tree, TreePart part, size_t dimPerNode, size_t numNodes) {
        /* Read data from files */
        m_prefix = tree.strOfPart(part) + m_prefix;
        m_d_sqrtQ = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(tree.path() + m_prefix + "sqrtQ" + tree.fpFileExt()));
        m_d_translation = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(tree.path() + m_prefix + "translation" + tree.fpFileExt()));
        m_dimPerNode = dimPerNode;
        m_numNodes = numNodes;
        DTensor<T> d_soc(this->m_dimPerNode, this->m_numNodes, 1);
        m_socs = std::make_unique<SocProjection<T>>(d_soc);
        m_dim = d_soc.numEl();
    }

    std::ostream &print(std::ostream &out) const {
        T t0 = (*m_d_translation)(0, 0, 0);
        bool linear = (t0 < 1e-6 && -t0 < 1e-6);
        if (linear) out << "Cost: Quadratic and linear\n";
        else out << "Cost: Quadratic\n";
        return out;
    }

public:
    ~Cost() = default;

    size_t dimPerNode() { return m_dimPerNode; }

    size_t dim() { return m_dim; }

    size_t numNodes() { return m_numNodes; }

    DTensor<T> &sqrtQ() { return *m_d_sqrtQ; }

    virtual DTensor<T> &sqrtR() { err << "[Cost::sqrtR] this is a leaf cost\n"; throw ERR; }

    void translate(DTensor<T> &d_vec) {
        d_vec += *m_d_translation;
    }

    void project(DTensor<T> &d_vec) {
        m_socs->project(d_vec);
    }

    friend std::ostream &operator<<(std::ostream &out, const Cost<T> &data) { return data.print(out); }
};


/**
 * Leaf costs
*/
template<typename T>
class CostLeaf : public Cost<T> {

public:
    explicit CostLeaf(ScenarioTree<T> &tree) :
        Cost<T>(tree, leaf, tree.numStates() + 2, tree.numLeafNodes()) {
    }
};


/**
 * Nonleaf costs
 */
template<typename T>
class CostNonleaf : public Cost<T> {

protected:
    std::unique_ptr<DTensor<T>> m_d_sqrtR = nullptr;

public:
    explicit CostNonleaf(ScenarioTree<T> &tree) :
        Cost<T>(tree, nonleaf, tree.numStatesAndInputs() + 2, tree.numNodes()) {
        m_d_sqrtR = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(tree.path() + this->m_prefix + "sqrtR" + tree.fpFileExt()));
    }

    DTensor<T> &sqrtR() { return *m_d_sqrtR; }
};


#endif
