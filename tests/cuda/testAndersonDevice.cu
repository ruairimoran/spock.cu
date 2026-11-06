#include <gtest/gtest.h>
#include "anderson.cuh"
#include "referenceAndersonCpu.cuh"
#include <array>
#include <limits>

namespace {
template<typename T>
class AndersonDeviceTest : public testing::Test {};
using Precisions = testing::Types<float, double>;
TYPED_TEST_SUITE(AndersonDeviceTest, Precisions);

template<typename T>
void compareFits(const AndersonFitInfo &expected, const AndersonFitInfo &actual,
                 DTensor<T> &cpuDirection, DTensor<T> &gpuDirection, DTensor<T> &gamma) {
    ASSERT_EQ(expected.status, actual.status);
    EXPECT_EQ(expected.selectedColumns, actual.selectedColumns);
    const double tolerance = std::is_same_v<T, float> ? 3e-4 : 2e-11;
    EXPECT_NEAR(expected.scale, actual.scale, tolerance * std::max(1., std::abs(expected.scale)));
    EXPECT_NEAR(expected.coefficientNorm, actual.coefficientNorm,
                tolerance * std::max(1., expected.coefficientNorm));
    EXPECT_NEAR(expected.directionRatio, actual.directionRatio,
                tolerance * std::max(1., expected.directionRatio));
    std::vector<T> expectedDirection, actualDirection, deviceGamma;
    cpuDirection.download(expectedDirection); gpuDirection.download(actualDirection); gamma.download(deviceGamma);
    for (size_t col = 0; col < expected.coefficients.size(); ++col) {
        const double allowed = tolerance * std::max(1., std::abs(expected.coefficients[col]));
        EXPECT_NEAR(expected.coefficients[col], actual.coefficients[col], allowed);
        EXPECT_NEAR(expected.coefficients[col], deviceGamma[col], allowed);
    }
    double scale = 0., error = 0.;
    for (size_t row = 0; row < expectedDirection.size(); ++row) {
        scale = std::hypot(scale, double(expectedDirection[row]));
        error = std::hypot(error, double(actualDirection[row]) - double(expectedDirection[row]));
    }
    EXPECT_LE(error, tolerance * std::max(scale, std::numeric_limits<double>::min()));
}

TYPED_TEST(AndersonDeviceTest, matchesCpuReferenceAcrossScalesRanksAndRejections) {
    using T = TypeParam;
    const auto before = Session::getInstance().totalAllocatedBytes();
    for (size_t stream : std::array<size_t,2>{0,1}) {
        if (stream >= Session::getInstance().numStreams()) continue;
        for (auto backend : {AndersonFitBackend::gpu,AndersonFitBackend::cpu})
        for (bool rolling : {false,true}) for (size_t memory : {1,3,8}) for (int scenario = 0; scenario < 9; ++scenario) {
            SCOPED_TRACE(::testing::Message() << stream << '/' << rolling << '/' << memory << '/' << scenario);
            constexpr size_t rows = 65;
            std::vector<T> c(rows*memory), y(rows*memory), r(rows);
            const double tiny = std::is_same_v<T,float> ? 1e-12 : 1e-120;
            const double large = std::is_same_v<T,float> ? 1e12 : 1e120;
            const double scale = scenario == 3 ? tiny : (scenario == 4 ? large : 1.);
            for (size_t row = 0; row < rows; ++row) {
                r[row] = T(scale * std::cos(.23*row));
                for (size_t col = 0; col < memory; ++col) {
                    y[row+col*rows] = T(scale * (std::sin(.17*(row+1)*(col+1)) + (row == col ? 2. : 0.)));
                    c[row+col*rows] = T(.2 * double(y[row+col*rows]));
                }
            }
            if (scenario == 1 && memory > 1)
                for (size_t row = 0; row < rows; ++row) y[row+(memory-1)*rows] = y[row];
            if (scenario == 2) std::fill(y.begin(),y.end(),T(0));
            if (scenario == 7) c[0] = std::numeric_limits<T>::infinity();
            if (scenario == 8) y[0] = std::numeric_limits<T>::quiet_NaN();
            DTensor<T> dC(c,rows,memory), dY(y,rows,memory), residual(r,rows);
            DTensor<T> q(rows,memory), rhs(rows), direction(rows);
            DTensor<T> oldQ(rows,memory), oldRhs(rows), oldDirection(rows);
            // Match Cache's actual borrowed layout: coefficients overwrite the
            // projected RHS prefix after that RHS has been consumed by the fit.
            DTensor<T> gamma(rhs,0,0,memory-1), oldGamma(oldRhs,0,0,memory-1);
            for (auto *tensor : {&dC,&dY,&residual,&q,&rhs,&gamma,&direction,&oldQ,&oldRhs,&oldGamma,&oldDirection})
                tensor->setStreamIdx(stream);
            AndersonOptions options;
            options.qr.incremental = rolling;
            options.fitBackend = backend;
            if (scenario == 5) options.maxCoefficientNorm = 1e-12;
            if (scenario == 6) options.maxDirectionRatio = 1e-6;
            RegularisedAnderson<T> gpu(dC,dY,q,rhs,gamma,options);
            CpuReferenceAnderson<T> cpu(dC,dY,oldQ,oldRhs,oldGamma,options);
            const auto allocated = Session::getInstance().totalAllocatedBytes();
            for (int repeat = 0; repeat < 2; ++repeat) {
                cpu.reset(); gpu.reset();
                compareFits(cpu.compute(residual,oldDirection),gpu.compute(residual,direction),oldDirection,direction,gamma);
                EXPECT_EQ(allocated,Session::getInstance().totalAllocatedBytes());
            }
        }
    }
    EXPECT_EQ(before,Session::getInstance().totalAllocatedBytes());
}

TYPED_TEST(AndersonDeviceTest, statefulUpdatesReusesAndResetsMatchCpu) {
    using T = TypeParam;
    constexpr size_t rows = 65, memory = 3;
    DTensor<T> c(rows,memory), y(rows,memory), residual(rows), q(rows,memory), oldQ(rows,memory);
    DTensor<T> rhs(rows), oldRhs(rows), direction(rows), oldDirection(rows);
    DTensor<T> gamma(rhs,0,0,memory-1), oldGamma(oldRhs,0,0,memory-1);
    const size_t stream = Session::getInstance().numStreams()-1;
    for (auto *tensor : {&c,&y,&residual,&q,&oldQ,&rhs,&oldRhs,&gamma,&oldGamma,&direction,&oldDirection})
        tensor->setStreamIdx(stream);
    AndersonOptions options;
    options.qr.rebuildInterval = 5;
    options.qr.checkInterval = 2;
    RegularisedAnderson<T> gpu(c,y,q,rhs,gamma,options);
    CpuReferenceAnderson<T> cpu(c,y,oldQ,oldRhs,oldGamma,options);
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    for (size_t step = 0; step < 35; ++step) {
        std::vector<T> hostC(rows*memory), hostY(rows*memory), r(rows);
        for (size_t row = 0; row < rows; ++row) {
            r[row] = T(std::cos(.31*row + .01*step));
            for (size_t col = 0; col < memory; ++col) {
                const size_t stamp = step+memory-col;
                hostY[row+col*rows] = T(std::sin(.17*(row+1)*(stamp+1)) + (row == stamp%rows ? 2. : 0.));
                hostC[row+col*rows] = T(.2*double(hostY[row+col*rows]));
            }
        }
        c.upload(hostC); y.upload(hostY); residual.upload(r);
        if (step == 17) { cpu.reset(); gpu.reset(); }
        cpu.historyChanged(); gpu.historyChanged();
        compareFits(cpu.compute(residual,oldDirection),gpu.compute(residual,direction),oldDirection,direction,gamma);
        if (step%3 == 0)
            compareFits(cpu.compute(residual,oldDirection),gpu.compute(residual,direction),oldDirection,direction,gamma);
        EXPECT_EQ(cpu.qrStatistics().rebuilds,gpu.qrStatistics().rebuilds);
        EXPECT_EQ(cpu.qrStatistics().updates,gpu.qrStatistics().updates);
        EXPECT_EQ(cpu.qrStatistics().reuses,gpu.qrStatistics().reuses);
        EXPECT_EQ(allocated,Session::getInstance().totalAllocatedBytes());
    }
}

TYPED_TEST(AndersonDeviceTest, deviceStatusChecksDoNotConsumeFailedSvdOutputs) {
    using T = TypeParam;
    using namespace anderson_reduced_detail;
    constexpr int memory = 2;
    DTensor<T> u(2,2), vt(2,2), singularValues(2), gamma(2,1,1,true);
    DTensor<double> rhs(2), solution(2);
    DTensor<int> info(1), indices(std::vector<int>{0,1},2);
    DTensor<FitState> state(1);
    singularValues.upload({std::numeric_limits<T>::quiet_NaN(),T(1)});
    const auto stream = spock::executionStream(gamma);
    for (int scenario = 0; scenario < 3; ++scenario) {
        const auto initialStatus = scenario == 2 ? AndersonFitStatus::nonfiniteHistory : AndersonFitStatus::accepted;
        state.upload(std::vector<FitState>{{initialStatus,2,1.,0.,0.}});
        info.upload(std::vector<int>{scenario == 0 ? 1 : 0});
        coefficients<<<1,1,0,stream>>>(u.raw(),vt.raw(),singularValues.raw(),info.raw(),rhs.raw(),indices.raw(),
            memory,1e-8,1e6,solution.raw(),gamma.raw(),state.raw());
        gpuErrChk(cudaGetLastError());
        std::vector<FitState> result;
        state.download(result);
        const auto expected = scenario == 0 ? AndersonFitStatus::solverFailure :
            (scenario == 1 ? AndersonFitStatus::nonfiniteSolution : AndersonFitStatus::nonfiniteHistory);
        EXPECT_EQ(expected,result[0].status);
        std::vector<T> values;
        gamma.download(values);
        EXPECT_EQ((std::vector<T>{0,0}),values);
    }
}

TYPED_TEST(AndersonDeviceTest, deviceDirectionNormHandlesExtremeAndNonfiniteValues) {
    using T = TypeParam;
    constexpr size_t rows = 4097;
    DTensor<T> gamma(1), rhs(std::vector<T>{0},1), residual(std::vector<T>(rows,T(1)),rows), direction(rows);
    AndersonDeviceFit<T> fit(1,rows,gamma);
    AndersonFitInfo info;
    info.selectedColumns.reserve(1); info.coefficients.resize(1);
    const auto allocated = Session::getInstance().totalAllocatedBytes();
    const double tiny = std::is_same_v<T,float> ? 1e-12 : 1e-120;
    const double large = std::is_same_v<T,float> ? 1e12 : 1e120;
    for (T value : {T(0),T(tiny),T(large),std::numeric_limits<T>::denorm_min()*T(8),
                    std::numeric_limits<T>::quiet_NaN(),std::numeric_limits<T>::infinity(),std::numeric_limits<T>::max()/T(4)}) {
        fit.matrix().upload(std::vector<T>{1});
        direction.upload(std::vector<T>(rows,value));
        fit.solve(rhs,1e-8,1e-8,1e6);
        const double expectedNorm = T(double(std::abs(value))*std::sqrt(double(rows)));
        fit.finish(residual,direction,1.,std::numeric_limits<double>::max(),info);
        if (std::isfinite(expectedNorm)) {
            EXPECT_TRUE(info.accepted());
            EXPECT_NEAR(expectedNorm,info.directionRatio,
                (std::is_same_v<T,float> ? 2e-6 : 2e-12)*std::max(expectedNorm,std::numeric_limits<double>::min()));
        } else {
            EXPECT_EQ(AndersonFitStatus::nonfiniteSolution,info.status);
            std::vector<T> actual;
            direction.download(actual);
            for (T element : actual) EXPECT_EQ(T(-1),element);
        }
        EXPECT_EQ(allocated,Session::getInstance().totalAllocatedBytes());
    }
}
} // namespace
