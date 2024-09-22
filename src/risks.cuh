#ifndef RISKS_CUH
#define RISKS_CUH

#include "../include/gpu.cuh"
#include "cones.cuh"


/**
 * Base class for a coherent risk
 * that can be described by the tuple
 * (E, F, K, b) = (matrix, matrix, cone, vector)
 */
template<typename T>
class CoherentRisk {

protected:
    std::unique_ptr<DTensor<T>> m_d_nullspaceProjectionMatrix = nullptr;
    std::unique_ptr<Cartesian<T>> m_K = nullptr;
    std::unique_ptr<DTensor<T>> m_d_b = nullptr;

    explicit CoherentRisk(size_t nodeIdx, DTensor<T> &nullspaceProj, DTensor<T> &b) {
        m_d_nullspaceProjectionMatrix = std::make_unique<DTensor<T>>(nullspaceProj, 2, nodeIdx, nodeIdx);
        m_d_b = std::make_unique<DTensor<T>>(b, 2, nodeIdx, nodeIdx);
    }

    virtual std::ostream &print(std::ostream &out) const { return out; };

public:
    virtual ~CoherentRisk() {}

    virtual DTensor<T> &nullspace() { return *m_d_nullspaceProjectionMatrix; }

    virtual Cartesian<T> &cone() { return *m_K; }

    virtual DTensor<T> &b() { return *m_d_b; }

    friend std::ostream &operator<<(std::ostream &out, const CoherentRisk<T> &data) { return data.print(out); }
};


/**
 * Average Value at Risk (AVaR)
*/
template<typename T>
class AVaR : public CoherentRisk<T> {

protected:
    std::unique_ptr<NonnegativeOrthantCone<T>> m_nnoc = nullptr;
    std::unique_ptr<ZeroCone<T>> m_zero = nullptr;

    std::ostream &print(std::ostream &out) const {
        out << "Risk: AVaR, \n";
        out << this->m_K;
        return out;
    }

public:
    explicit AVaR(size_t nodeIdx, size_t numChildren, DTensor<T> &nullspaceProj, DTensor<T> &b) :
        CoherentRisk<T>(nodeIdx, nullspaceProj, b) {
        size_t doubleNumCh = numChildren * 2;
        /* The zero cone is extended for nodes with fewer children.
         * As we are projecting on the dual, the buffer elements will never be changed.
         */
        size_t fill = this->m_d_b->numEl() - doubleNumCh;
        m_nnoc = std::make_unique<NonnegativeOrthantCone<T>>(doubleNumCh);
        m_zero = std::make_unique<ZeroCone<T>>(fill);
        this->m_K = std::make_unique<Cartesian<T>>();
        this->m_K->addCone(*m_nnoc);
        this->m_K->addCone(*m_zero);
    }
};


#endif
