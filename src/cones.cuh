#ifndef CONES_CUH
#define CONES_CUH

#include "../include/gpu.cuh"

TEMPLATE_WITH_TYPE_T
__global__ void k_maxWithZero(T *vec, size_t n);

TEMPLATE_WITH_TYPE_T
__global__ void k_setToZero(T *vec, size_t n);

TEMPLATE_WITH_TYPE_T
__global__ void k_projectOnSoc(T *vec, size_t n, T nrm, T scaling);


TEMPLATE_WITH_TYPE_T
class ConvexCone {

protected:
    size_t m_dimension = 0;
    size_t m_axis = 0;  // Do not change! 0=rows, 1=cols, 2=mats

    explicit ConvexCone(size_t dim) : m_dimension(dim) {}

    bool dimensionCheck(DTensor<T> &d_vec) {
        if (d_vec.numRows() != m_dimension || d_vec.numCols() != 1 || d_vec.numMats() != 1) {
            std::cerr << "DTensor is [" << d_vec.numRows() << " x " << d_vec.numCols() << " x " << d_vec.numMats()
                      << "], but cone has dimensions [" << m_dimension << " x " << 1 << " x " << 1 << "]\n";
            throw std::invalid_argument("DTensor and cone dimensions mismatch");
        }
        return true;
    }

public:
    virtual ~ConvexCone() {}

    size_t dimension() { return m_dimension; }

    virtual std::string name() = 0;

    virtual void project(DTensor<T> &d_vec) = 0;

    virtual void projectOnDual(DTensor<T> &d_vec) = 0;

};


/**
 * A null cone (Null)
 * - used as placeholder
*/
TEMPLATE_WITH_TYPE_T
class NullCone : public ConvexCone<T> {

public:
    explicit NullCone(size_t dim) : ConvexCone<T>(dim) {}

    void project(DTensor<T> &d_vec) {
        throw std::invalid_argument("Cannot project on null cone");
    }

    void projectOnDual(DTensor<T> &d_vec) {
        project(d_vec);
    }

    std::string name() { return "Null cone"; }

};


/**
 * The Universe cone (Univ)
 * - the set is R^n
 * - the dual is the Zero cone
*/
TEMPLATE_WITH_TYPE_T
class UniverseCone : public ConvexCone<T> {

public:
    explicit UniverseCone(size_t dim) : ConvexCone<T>(dim) {}

    void project(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        return;  // Do nothing!
    }

    void projectOnDual(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_setToZero<<<DIM2BLOCKS(this->m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), this->m_dimension);
    }

    std::string name() { return "Universe cone"; }

};


/**
 * The Zero cone (Zero)
 * - the set is {0}
 * - the dual is the Universe cone
*/
TEMPLATE_WITH_TYPE_T
class ZeroCone : public ConvexCone<T> {

public:
    explicit ZeroCone(size_t dim) : ConvexCone<T>(dim) {}

    void project(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_setToZero<<<DIM2BLOCKS(this->m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), this->m_dimension);
    }

    void projectOnDual(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        return;  // Do nothing!
    }

    std::string name() { return "Zero cone"; }

};


/**
 * The Nonnegative Orthant cone (NnOC)
 * - the set is R^n_+
 * - the cone is self dual
*/
TEMPLATE_WITH_TYPE_T
class NonnegativeOrthantCone : public ConvexCone<T> {

public:
    explicit NonnegativeOrthantCone(size_t dim) : ConvexCone<T>(dim) {}

    void project(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        k_maxWithZero<<<DIM2BLOCKS(this->m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), this->m_dimension);
    }

    void projectOnDual(DTensor<T> &d_vec) {
        project(d_vec);
    }

    std::string name() { return "Nonnegative Orthant cone"; }

};


/**
 * The Second Order cone (SOC)
 * - the set is R^n_2
 * - the cone is self dual
 * - this projection follows [page 184, Section 6.3.2] of
 * > Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and trends® in Optimization, 1(3), 127-239.
*/
TEMPLATE_WITH_TYPE_T
class SecondOrderCone : public ConvexCone<T> {

public:
    explicit SecondOrderCone(size_t dim) : ConvexCone<T>(dim) {}

    void project(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        /* Determine the 2-norm of the first (n - 1) elements of d_vec */
        DTensor<T> vecFirstPart(d_vec, this->m_axis, 0, this->m_dimension - 2);
        T nrm = vecFirstPart.normF();
        float vecLastElement = d_vec(this->m_dimension - 1);
        if (nrm <= vecLastElement) {
            return;  // Do nothing!
        } else if (nrm <= -vecLastElement) {
            k_setToZero<<<DIM2BLOCKS(this->m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), this->m_dimension);
        } else {
            T scaling = (nrm + vecLastElement) / (2. * nrm);
            k_projectOnSoc<<<DIM2BLOCKS(this->m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), this->m_dimension, nrm, scaling);
        }
    }

    void projectOnDual(DTensor<T> &d_vec) {
        project(d_vec);
    }

    std::string name() { return "Second Order cone"; }

};


/**
 * A Cartesian cone (Cart)
 * - the set is a Cartesian product of cones (cone x cone x ...)
 * - the dual is the concatenation of the dual of each constituent cone 
*/
TEMPLATE_WITH_TYPE_T
class Cartesian : public ConvexCone<T> {

private:
    std::vector<ConvexCone<T> *> m_cones;

public:
    explicit Cartesian() : ConvexCone<T>(0) {}

    void addCone(ConvexCone<T> &cone) {
        m_cones.push_back(&cone);
        this->m_dimension += cone.dimension();
    }

    void project(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        size_t start = 0;
        for (ConvexCone<T> *set: m_cones) {
            size_t coneDim = set->dimension();
            size_t end = start + coneDim - 1;
            DTensor<T> vecSlice(d_vec, this->m_axis, start, end);
            set->project(vecSlice);
            start += coneDim;
        }
    }

    void projectOnDual(DTensor<T> &d_vec) {
        this->dimensionCheck(d_vec);
        size_t start = 0;
        for (ConvexCone<T> *set: m_cones) {
            size_t coneDim = set->dimension();
            size_t end = start + coneDim - 1;
            DTensor<T> vecSlice(d_vec, this->m_axis, start, end);
            set->projectOnDual(vecSlice);
            start += coneDim;
        }
    }

    std::string name() { return "Cartesian cone"; }

    void print() {
        std::cout << "Cartesian cone (" << this->m_dimension << ") of: ";
        for (ConvexCone<T> *cone: m_cones) {
            std::cout << "+ " << cone->name() << " (" << cone->dimension() << ") ";
        }
        std::cout << "\n";
    }
};

#endif
