# ---- cmake for CUDA with googleTest
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()
project(SPOCK
    DESCRIPTION "Solver for risk-averse optimal control problems."
    HOMEPAGE_URL "https://github.com/ruairimoran/raocp-parallel"
    LANGUAGES CXX)
# ----
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_FLAGS "-std=c++20")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++20)
enable_language(CUDA)
# ----
add_library(spock_compiler_flags INTERFACE)
target_compile_features(spock_compiler_flags INTERFACE cxx_std_20)
set(CMAKE_CXX_EXTENSIONS OFF)
# ----
add_library(spock_developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -std=c++20 -Xcompiler=-Wall -Xcudafe=--display_error_number -g)
target_compile_options(spock_developer_flags
    INTERFACE
    # flags for CXX builds
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    # flags for CUDA builds
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
target_link_libraries(spock_compiler_flags INTERFACE $<BUILD_INTERFACE:spock_developer_flags>
)
# ----
include(FetchContent)
FetchContent_Declare(
    gputils
    URL https://github.com/GPUEngineering/GPUtils/archive/refs/tags/v2.2.0.zip
    URL_HASH SHA256=eab172556dcca09b1f23e9c72724f4c83d7010fc0a12bd0a6144c2648c5b2993
    # SPOCK only consumes the headers. GPUtils' standalone CMake project
    # selects its own CUDA compiler, which can cause a reconfiguration loop.
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(gputils)
FetchContent_GetProperties(
    gputils
    SOURCE_DIR GPUTILS_SOURCE_DIR
)
# ----
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    SOURCE_SUBDIR Eigen
)
FetchContent_MakeAvailable(eigen)
# ----
add_subdirectory(src)
option(SPOCK_BUILD_TEST "Build SPOCK tests" OFF)
if (SPOCK_BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()
# ----
add_subdirectory(examples/precondition)
add_subdirectory(examples/andersonBuffer)
add_subdirectory(examples/random)
add_subdirectory(examples/serverAirCondition)
add_subdirectory(examples/networkedControl)
add_subdirectory(examples/powerDistribution)
add_subdirectory(examples/networkedControlMpc)
# ----
