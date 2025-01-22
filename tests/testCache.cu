#include <gtest/gtest.h>
#include "../src/cache.cuh"
#include <filesystem>
#include <iostream>


class CacheTest : public testing::Test {
protected:
    CacheTest() = default;
};

TEMPLATE_WITH_TYPE_T
class CacheTestData {
public:
    std::string m_path = "../../data/";
    std::unique_ptr<ScenarioTree<T>> m_tree;
    std::unique_ptr<ProblemData<T>> m_data;
    std::unique_ptr<Cache<T>> m_cache;

    /** Prepare some host and device data */
    bool m_detectInfeas = false;
    T m_tol = 1e-3;
    size_t m_maxIters = 1500;
    size_t m_maxInnerIters = 8;
    size_t m_andersonBuff = 3;
    bool m_allowK0 = true;
    bool m_debug = false;

    CacheTestData() {
        m_tree = std::make_unique<ScenarioTree<T>>(m_path);
        m_data = std::make_unique<ProblemData<T>>(*m_tree);
        m_cache = std::make_unique<Cache<T>>(*m_tree, *m_data, m_detectInfeas, m_tol, m_maxIters,
                                             m_maxInnerIters, m_andersonBuff, m_allowK0, m_debug);
    };

    virtual ~CacheTestData() = default;
};

/* ---------------------------------------
 * Initialise state
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testInitialisingState(CacheTestData<T> &d) {
    std::vector<T> initialState = {3., 5., 4.};
    d.m_cache->initialiseState(initialState);
    std::vector<T> x(initialState.size());
    d.m_cache->m_d_initState->download(x);
    EXPECT_EQ(x, initialState);
}

TEST_F(CacheTest, initialisingState) {
    CacheTestData<float> df;
    testInitialisingState<float>(df);
    CacheTestData<double> dd;
    testInitialisingState<double>(dd);
}

/* ---------------------------------------
 * Project on kernels (online)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testKernelProjectionOnline(CacheTestData<T> &d, T epsilon) {
    /* Create random tensor data to be projected */
    T hi = 100.;
    T lo = -hi;
    size_t matAxis = 2;
    for (size_t node = 0; node < d.m_tree->numNonleafNodes(); node++) {
        size_t chFr = d.m_tree->childFrom()[node];
        size_t chTo = d.m_tree->childTo()[node];
        size_t numCh = d.m_tree->numChildren()[node];
        size_t actualSizeY = numCh * 2 + 1;
        DTensor<T> yPadded(*(d.m_cache->m_d_y), matAxis, node, node);
        DTensor<T> y(yPadded, 0, 0, actualSizeY - 1);
        DTensor<T> t(*(d.m_cache->m_d_t), matAxis, chFr, chTo);
        DTensor<T> s(*(d.m_cache->m_d_s), matAxis, chFr, chTo);
        DTensor<T> randY = DTensor<T>::createRandomTensor(actualSizeY, 1, 1, lo, hi);
        DTensor<T> randT = DTensor<T>::createRandomTensor(1, 1, numCh, lo, hi);
        DTensor<T> randS = DTensor<T>::createRandomTensor(1, 1, numCh, lo, hi);
        randY.deviceCopyTo(y);
        randT.deviceCopyTo(t);
        randS.deviceCopyTo(s);
    }

    /* Project random data */
    d.m_cache->projectPrimalWorkspaceOnKernels();

    /* Check result is in kernel */
    for (size_t node = 0; node < d.m_tree->numNonleafNodes(); node++) {
        size_t chFr = d.m_tree->childFrom()[node];
        size_t chTo = d.m_tree->childTo()[node];
        size_t numCh = d.m_tree->numChildren()[node];
        size_t actualSizeY = numCh * 2 + 1;
        DTensor<T> yPadded(*(d.m_cache->m_d_y), matAxis, node, node);
        DTensor<T> y(yPadded, 0, 0, actualSizeY - 1);
        DTensor<T> t(*(d.m_cache->m_d_t), matAxis, chFr, chTo);
        DTensor<T> s(*(d.m_cache->m_d_s), matAxis, chFr, chTo);
        /* Copy in projected data */
        DTensor<T> projected(d.m_data->nullDim());
        DTensor<T> projY(projected, 0, 0, actualSizeY - 1);
        DTensor<T> projT(projected, 0, d.m_data->yDim(), d.m_data->yDim() + numCh - 1);
        DTensor<T> projS(projected, 0, d.m_data->yDim() + d.m_tree->numEvents(),
                         d.m_data->yDim() + d.m_tree->numEvents() + numCh - 1);
        projT.reshape(1, 1, numCh);
        projS.reshape(1, 1, numCh);
        y.deviceCopyTo(projY);
        t.deviceCopyTo(projT);
        s.deviceCopyTo(projS);
        /* Compute kernel matrix * projected vector */
        DTensor<T> kerConMats = DTensor<T>::parseFromFile(d.m_path + "S2" + d.m_tree->fpFileExt());
        DTensor<T> kerConMat(kerConMats, 2, node, node);
        DTensor<T> shouldBeZeros = kerConMat * projected;
        std::vector<T> result(shouldBeZeros.numEl());
        shouldBeZeros.download(result);
        /* Ensure result is zeros */
        for (size_t i = 0; i < shouldBeZeros.numEl(); i++) { EXPECT_NEAR(result[i], 0., epsilon); }
    }
}

TEST_F(CacheTest, kernelProjectionOnline) {
    CacheTestData<float> df;
    testKernelProjectionOnline<float>(df, TEST_PRECISION_LOW);
    CacheTestData<double> dd;
    testKernelProjectionOnline<double>(dd, TEST_PRECISION_HIGH);
}

/* ---------------------------------------
 * Project on kernels (online)
 * - Orthogonality test
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void buildVector(DTensor<T> &vec, size_t yAct, size_t yFull, size_t numCh, size_t numEvents,
                 DTensor<T> &y, DTensor<T> &t, DTensor<T> &s) {
    DTensor<T> vecY(vec, 0, 0, yAct - 1);
    DTensor<T> vecT(vec, 0, yFull, yFull + numCh - 1);
    DTensor<T> vecS(vec, 0, yFull + numEvents, yFull + numEvents + numCh - 1);
    vecT.reshape(1, 1, numCh);
    vecS.reshape(1, 1, numCh);
    y.deviceCopyTo(vecY);
    t.deviceCopyTo(vecT);
    s.deviceCopyTo(vecS);
}

TEMPLATE_WITH_TYPE_T
void testKernelProjectionOnlineOrthogonality(CacheTestData<T> &d, T epsilon) {
    /* Create original data */
    T hi = 100.;
    T lo = -hi;
    size_t matAxis = 2;
    size_t node = 0;
    size_t chFr = d.m_tree->childFrom()[node];
    size_t chTo = d.m_tree->childTo()[node];
    size_t numCh = d.m_tree->numChildren()[node];
    size_t actualSizeY = numCh * 2 + 1;
    DTensor<T> yPadded(*(d.m_cache->m_d_y), matAxis, node, node);
    DTensor<T> y(yPadded, 0, 0, actualSizeY - 1);
    DTensor<T> t(*(d.m_cache->m_d_t), matAxis, chFr, chTo);
    DTensor<T> s(*(d.m_cache->m_d_s), matAxis, chFr, chTo);
    DTensor<T> randY = DTensor<T>::createRandomTensor(actualSizeY, 1, 1, lo, hi);
    DTensor<T> randT = DTensor<T>::createRandomTensor(1, 1, numCh, lo, hi);
    DTensor<T> randS = DTensor<T>::createRandomTensor(1, 1, numCh, lo, hi);
    randY.deviceCopyTo(y);
    randT.deviceCopyTo(t);
    randS.deviceCopyTo(s);
    /* Build original data */
    DTensor<T> original(d.m_data->nullDim());
    buildVector(original, actualSizeY, d.m_data->yDim(), numCh, d.m_tree->numEvents(), y, t, s);
    /* Project original data */
    d.m_cache->projectPrimalWorkspaceOnKernels();
    /* Build projected data */
    DTensor<T> projected(d.m_data->nullDim());
    buildVector(projected, actualSizeY, d.m_data->yDim(), numCh, d.m_tree->numEvents(), y, t, s);
    /* Build other data */
    DTensor<T> other(d.m_data->nullDim());
    buildVector(other, actualSizeY, d.m_data->yDim(), numCh, d.m_tree->numEvents(), y, t, s);
    /* Project other data */
    d.m_cache->projectPrimalWorkspaceOnKernels();
    /* Build otherProjected data */
    DTensor<T> otherProjected(d.m_data->nullDim());
    buildVector(otherProjected, actualSizeY, d.m_data->yDim(), numCh, d.m_tree->numEvents(), y, t, s);
    /* Orthogonality test (otherProjected - projected) † (projected - original) */
    DTensor<T> a = otherProjected - projected;
    DTensor<T> b = projected - original;
    EXPECT_LT(a.dotF(b), epsilon);
}

TEST_F(CacheTest, kernelProjectionOnlineOrthogonality) {
    CacheTestData<float> df;
    testKernelProjectionOnlineOrthogonality<float>(df, TEST_PRECISION_LOW);
    CacheTestData<double> dd;
    testKernelProjectionOnlineOrthogonality<double>(dd, TEST_PRECISION_HIGH);
}

/* ---------------------------------------
 * Test dotM operator
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testDotM(CacheTestData<T> &d, T epsilon) {
    std::string ext = d.m_tree->fpFileExt();
    DTensor<T> dotVector = DTensor<T>::parseFromFile(d.m_path + "dotVector" + ext);
    DTensor<T> dotResult = DTensor<T>::parseFromFile(d.m_path + "dotResult" + ext);
    Cache<T> &c = *d.m_cache;
    T dot = c.dotM(dotVector, dotVector);
    std::vector<T> expected(1);
    dotResult.download(expected);
    EXPECT_NEAR(dot, expected[0], epsilon);
}

TEST_F(CacheTest, dotM) {
    CacheTestData<float> df;
    testDotM<float>(df, TEST_PRECISION_LOW);
    CacheTestData<double> dd;
    testDotM<double>(dd, TEST_PRECISION_HIGH);
}

/* ---------------------------------------
 * Data transfer T/S in
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyInTS(CacheTestData<T> &d) {
    size_t nR = 9, nC = 1, nM = 8;
    size_t dstStart = 6;
    DTensor<T> src = DTensor<T>::createRandomTensor(1, nC, nM, -10, 10);
    DTensor<T> dst(nR, nC, nM, true);
    std::vector<size_t> anc = {0, 0, 0, 1, 1, 2, 2, 2};
    DTensor<size_t> d_anc(anc, nM);
    std::vector<size_t> chFrom = {1, 3, 5};
    DTensor<size_t> d_chFrom(chFrom, 3);
    k_memCpyInTS<<<numBlocks(nM, TPB), TPB>>>(dst.raw(), src.raw(), nM, nR, dstStart, d_anc.raw(), d_chFrom.raw());
    std::vector<size_t> numCh = {2, 2, 3};
    for (size_t mat = 1; mat < nM; mat++) {
        size_t anc_ = anc[mat];
        size_t idx = mat - chFrom[anc_];
        EXPECT_EQ(src(0, 0, mat), dst(dstStart+idx, 0, anc_));
    }
}

TEST_F(CacheTest, memCpyInTS) {
    CacheTestData<float> df;
    testMemCpyInTS<float>(df);
    CacheTestData<double> dd;
    testMemCpyInTS<double>(dd);
}

/* ---------------------------------------
 * Data transfer T/S out
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testMemCpyOutTS(CacheTestData<T> &d) {
    size_t nR = 9, nC = 1, nM = 8;
    size_t srcStart = 6;
    DTensor<T> src = DTensor<T>::createRandomTensor(nR, nC, nM, -10, 10);
    DTensor<T> dst(1, nC, nM);
    std::vector<size_t> anc = {0, 0, 0, 1, 1, 2, 2, 2};
    DTensor<size_t> d_anc(anc, nM);
    std::vector<size_t> chFrom = {1, 3, 5};
    DTensor<size_t> d_chFrom(chFrom, 3);
    k_memCpyOutTS<<<numBlocks(nM, TPB), TPB>>>(dst.raw(), src.raw(), nM, nR, srcStart, d_anc.raw(), d_chFrom.raw());
    std::vector<size_t> numCh = {2, 2, 3};
    for (size_t mat = 1; mat < nM; mat++) {
        size_t anc_ = anc[mat];
        size_t idx = mat - chFrom[anc_];
        EXPECT_EQ(dst(0, 0, mat), src(srcStart+idx, 0, anc_));
    }
}

TEST_F(CacheTest, memCpyOutTS) {
    CacheTestData<float> df;
    testMemCpyOutTS<float>(df);
    CacheTestData<double> dd;
    testMemCpyOutTS<double>(dd);
}

/* ---------------------------------------
 * Convergence (CP)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testConvergenceCP(CacheTestData<T> &d) {
    std::vector<T> initState(d.m_tree->numStates(), .1);
    int status = d.m_cache->runCp(initState);
    EXPECT_EQ(status, 0);
}

TEST_F(CacheTest, testConvergenceCP) {
    CacheTestData<float> df;
    testConvergenceCP<float>(df);
    CacheTestData<double> dd;
    testConvergenceCP<double>(dd);
}

/* ---------------------------------------
 * Convergence (SPOCK)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void testConvergenceSP(CacheTestData<T> &d) {
    std::vector<T> initState(d.m_tree->numStates(), .1);
    int status = d.m_cache->runSpock(initState);
    EXPECT_EQ(status, 0);
}

TEST_F(CacheTest, testConvergenceSP) {
    CacheTestData<float> df;
    testConvergenceSP<float>(df);
    CacheTestData<double> dd;
    testConvergenceSP<double>(dd);
}
