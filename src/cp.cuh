#ifndef CP_CUH
#define CP_CUH
#include "../include/gpu.cuh"
#include "cache.cuh"
#include <chrono>
#include <numeric>  // For debugging only


/**
 * Vanilla Chambolle-Pock (cp) algorithm with a parallelised cache
 */
size_t timeCp(Cache& cache, std::vector<real_t> initialState) {
    /* For debugging only ----------------- */
    std::vector<real_t> numbers(cache.solutionSize());
    std::iota (std::begin(numbers), std::end(numbers), 0);
    /* ------------------------------------ */
    std::cout << "timer started" << "\n";
    const auto tick = std::chrono::high_resolution_clock::now();
//    cache.vanillaCp(initialState, &numbers);  // For debugging only!
    cache.vanillaCp(initialState);  ///< run vanilla cp algorithm
    const auto tock = std::chrono::high_resolution_clock::now();
    auto durationMilli = std::chrono::duration<double, std::milli>(tock - tick).count();
    std::cout << "timer stopped:  " << durationMilli << " ms" << "\n";
    return 0;
}

#endif
