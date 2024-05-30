#ifndef GPU__
#define GPU__

#include <tensor.cuh>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <source_location>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


/**
 * Define defaults
 */
#define DEFAULT_FPX double
#define THREADS_PER_BLOCK_ 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))
#define DIM2BLOCKS_(n, t) ((n) / t + ((n) % t != 0))
#if (__cplusplus >= 201703L)  ///< if c++17 or above
#define TEMPLATE_WITH_TYPE_T template<typename T = DEFAULT_FPX>
#else
#define TEMPLATE_WITH_TYPE_T template<typename T>
#endif
#if (__cplusplus >= 202002L)  ///< if c++20 or above
#define TEMPLATE_CONSTRAINT_REQUIRES_FPX requires std::floating_point<T>
#else
#define TEMPLATE_CONSTRAINT_REQUIRES_FPX
#endif

/**
 * Debugging
 */
template<typename T>
static void printIfTensor(std::string description, std::unique_ptr<DTensor<T>> &data) {
    if (data) {
        std::cout << description << *data;
    } else {
        std::cout << description << "NOTHING TO PRINT.\n";
    }
}

#endif
