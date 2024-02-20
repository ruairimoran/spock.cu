#ifndef __CHAMBOLLE_POCK__
#define __CHAMBOLLE_POCK__
#include "../include/stdgpu.h"
#include "cache.cuh"
#include <chrono>


/**
 * Vanilla Chambolle-Pock (cp) algorithm with a parallelised cache
 */
size_t cp(Cache& cache, std::vector<real_t> initialState) {
    cache.initialiseState(initialState);
    size_t iter = 0;
    std::cout << "timer started" << std::endl;
    const auto tick = std::chrono::high_resolution_clock::now();
    for (iter=0; iter<cache.maxIters(); iter++) {
        /** compute one iteration of chock operator */
//        cache.chock();

        /** calculate current errors */
//        cache.computeErrors();

        /** check stopping criteria */
//        if (cache.error() <= tol) break;
    }
    const auto tock = std::chrono::high_resolution_clock::now();
    auto durationMilli = std::chrono::duration<double, std::milli>(tock - tick).count();
    std::cout << "timer stopped in " << durationMilli << " ms" << std::endl;
    cache.countOperations() = iter;

    return 0;
}

#endif
