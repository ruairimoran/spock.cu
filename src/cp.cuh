#ifndef __CHAMBOLLE_POCK__
#define __CHAMBOLLE_POCK__
#include "../include/stdgpu.h"
#include "cache.cuh"


/**
 * Vanilla Chambolle-Pock (cp) algorithm
 */
size_t cp(Cache& cache) {
    size_t i = 1;
    std::cout << "hello from cp: " << i << std::endl;
    return i;
}

#endif
