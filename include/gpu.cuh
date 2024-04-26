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

#define real_t double
#define REAL_PRECISION 1e-12

/** Debugging */
template<typename T>
static void printIf(std::string description, std::unique_ptr<DTensor<T>> &data) {
    if (data) {
        std::cout << description << *data;
    } else {
        std::cout << description << "NOTHING TO PRINT.";
    }
}

#endif
