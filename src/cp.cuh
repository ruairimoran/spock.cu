#ifndef __CHAMBOLLE_POCK__
#define __CHAMBOLLE_POCK__
#include "../include/stdgpu.h"
#include "cache.cuh"


/**
 * Vanilla Chambolle-Pock (cp) algorithm with a parallelised cache
 */
size_t cp(Cache& cache, std::vector<real_t> initialState) {
//    cache.cache_initial_state(initialState);
    size_t iter = 0;
    printf("timer started");
//    tick = time.perf_counter();
    for (iter=0; iter<cache.maxIters(); iter++) {
        /** compute one iteration of chock operator */
//        cache.chock();

        /** calculate current errors */
//        cache.computeErrors();

        /** check stopping criteria */
//        if (cache.error() <= tol) break;
    }

//    tock = time.perf_counter();
    printf("timer stopped in {tock - tick:0.4f} seconds");
    cache.countOperations() = iter;

    return 0;
}

#endif
