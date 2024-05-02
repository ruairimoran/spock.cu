#ifndef CONES_CUH
#define CONES_CUH

#include "../include/gpu.cuh"


__global__ void d_maxWithZero(real_t *vec, size_t n);

__global__ void d_setToZero(real_t *vec, size_t n);

__global__ void d_projectOnSoc(real_t *vec, size_t n, real_t nrm, real_t scaling);


class ConvexCone {

protected:
    size_t m_dimension = 0;
    size_t m_axis = 0;  // Do not change! 0=rows, 1=cols, 2=mats

    explicit ConvexCone(size_t dim) : m_dimension(dim) {}

    bool dimensionCheck(DTensor<real_t> &d_vec) {
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

    virtual void project(DTensor<real_t> &d_vec) = 0;

    virtual void projectOnDual(DTensor<real_t> &d_vec) = 0;

};


/**
 * A null cone (Null)
 * - used as placeholder
*/
class NullCone : public ConvexCone {

public:
    explicit NullCone(size_t dim) : ConvexCone(dim) {}

    void project(DTensor<real_t> &d_vec) {
        throw std::invalid_argument("Cannot project on null cone");
    }

    void projectOnDual(DTensor<real_t> &d_vec) {
        project(d_vec);
    }

    std::string name() { return "Null cone"; }

};


/**
 * The Universe cone (Univ)
 * - the set is R^n
 * - the dual is the Zero cone
*/
class UniverseCone : public ConvexCone {

public:
    explicit UniverseCone(size_t dim) : ConvexCone(dim) {}

    void project(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        return;  // Do nothing!
    }

    void projectOnDual(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        d_setToZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), m_dimension);
    }

    std::string name() { return "Universe cone"; }

};


/**
 * The Zero cone (Zero)
 * - the set is {0}
 * - the dual is the Universe cone
*/
class ZeroCone : public ConvexCone {

public:
    explicit ZeroCone(size_t dim) : ConvexCone(dim) {}

    void project(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        d_setToZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), m_dimension);
    }

    void projectOnDual(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        return;  // Do nothing!
    }

    std::string name() { return "Zero cone"; }

};


/**
 * The Nonnegative Orthant cone (NnOC)
 * - the set is R^n_+
 * - the cone is self dual
*/
class NonnegativeOrthantCone : public ConvexCone {

public:
    explicit NonnegativeOrthantCone(size_t dim) : ConvexCone(dim) {}

    void project(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        d_maxWithZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), m_dimension);
    }

    void projectOnDual(DTensor<real_t> &d_vec) {
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
class SecondOrderCone : public ConvexCone {

public:
    explicit SecondOrderCone(size_t dim) : ConvexCone(dim) {}

    void project(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        /** Determine the 2-norm of the first (n - 1) elements of d_vec */
        DTensor<real_t> vecFirstPart(d_vec, m_axis, 0, m_dimension - 2);
        real_t nrm = vecFirstPart.normF();
        float vecLastElement = d_vec(m_dimension - 1);
        if (nrm <= vecLastElement) {
            return;  // Do nothing!
        } else if (nrm <= -vecLastElement) {
            d_setToZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), m_dimension);
        } else {
            real_t scaling = (nrm + vecLastElement) / (2. * nrm);
            d_projectOnSoc<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(d_vec.raw(), m_dimension, nrm, scaling);
        }
    }

    void projectOnDual(DTensor<real_t> &d_vec) {
        project(d_vec);
    }

    std::string name() { return "Second Order cone"; }

};


/**
 * A Cartesian cone (Cart)
 * - the set is a Cartesian product of cones (cone x cone x ...)
 * - the dual is the concatenation of the dual of each constituent cone 
*/
class Cartesian : public ConvexCone {

private:
    std::vector<ConvexCone *> m_cones;

public:
    explicit Cartesian() : ConvexCone(0) {}

    void addCone(ConvexCone &cone) {
        m_cones.push_back(&cone);
        m_dimension += cone.dimension();
    }

    void project(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        size_t start = 0;
        for (ConvexCone *set: m_cones) {
            size_t coneDim = set->dimension();
            size_t end = start + coneDim - 1;
            DTensor<real_t> vecSlice(d_vec, m_axis, start, end);
            set->project(vecSlice);
            start += coneDim;
        }
    }

    void projectOnDual(DTensor<real_t> &d_vec) {
        dimensionCheck(d_vec);
        size_t start = 0;
        for (ConvexCone *set: m_cones) {
            size_t coneDim = set->dimension();
            size_t end = start + coneDim - 1;
            DTensor<real_t> vecSlice(d_vec, m_axis, start, end);
            set->projectOnDual(vecSlice);
            start += coneDim;
        }
    }

    std::string name() { return "Cartesian cone"; }

    void print() {
        std::cout << "Cartesian cone (" << m_dimension << ") of:\n";
        for (ConvexCone *cone: m_cones) {
            std::cout << "+ " << cone->name() << " (" << cone->dimension() << ")\n";
        }
    }
};

#endif
