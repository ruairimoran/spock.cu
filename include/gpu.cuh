#ifndef GPU__
#define GPU__

#include <tensor.cuh>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <source_location>
#include <sstream>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


/**
 * Define defaults
 */
#define TPB 512
#define TEST_PRECISION_LOW 1e-3
#define TEST_PRECISION_HIGH 1e-4
static std::ostringstream err;

/**
 * Debugging
 */
template<typename T>
static void printIfTensor(const std::string &description, std::unique_ptr<DTensor<T>> &data) {
    if (data) {
        std::cout << description << *data;
    } else {
        std::cout << description << "NOTHING TO PRINT.\n";
    }
}

#endif
