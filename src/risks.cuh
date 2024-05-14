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
    std::unique_ptr<Cartesian> m_K = nullptr;

    explicit CoherentRisk(size_t nodeIdx, DTensor<T> &nullspaceProj) {
        m_d_nullspaceProjectionMatrix = std::make_unique<DTensor<T>>(nullspaceProj, 2, nodeIdx, nodeIdx);
    }

public:
    virtual ~CoherentRisk() {}

    virtual DTensor<T> &nullspace() { return *m_d_nullspaceProjectionMatrix; }

    virtual Cartesian &cone() { return *m_K; }

    virtual void print() = 0;
};


/**
 * Average Value at Risk (AVaR)
*/
template<typename T>
class AVaR : public CoherentRisk<T> {

protected:
    NonnegativeOrthantCone m_nnoc;
    ZeroCone m_zero;

public:
    explicit AVaR(size_t nodeIdx, size_t numChildren, DTensor<T> &nullspaceProj) :
        CoherentRisk<T>(nodeIdx, nullspaceProj),
        m_nnoc(numChildren * 2),
        m_zero(1) {
        CoherentRisk<T>::m_K = std::make_unique<Cartesian>();
        CoherentRisk<T>::m_K->addCone(m_nnoc);
        CoherentRisk<T>::m_K->addCone(m_zero);
    }

    void print() {
        std::cout << "Risk: AVaR, \n";
        std::cout << "K: ";
        if (CoherentRisk<T>::m_K) {
            CoherentRisk<T>::m_K->print();
        } else {
            std::cout << "NO CONE TO PRINT.\n";
        }
    }

};


#endif
