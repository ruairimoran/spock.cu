# ---- cmake
cmake_minimum_required(VERSION 3.14)
project(raocp-parallel
    DESCRIPTION "Solver for risk-averse optimal control problems."
    HOMEPAGE_URL "https://github.com/ruairimoran/raocp-parallel"
    LANGUAGES CUDA CXX)

# ---- GoogleTest with nvcc (CUDA)
# requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
    GIT_TAG b796f7d44681514f58a683a3a71ff17c94edb0c1 # version 1.13.0
)
# for Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include_directories(. include ${CUDAToolkit_INCLUDE_DIRS})
cuda_add_executable(
    test_tree
    test_tree.cc
)
target_link_libraries(
    GTest::gtest_main
    CUDA::cublas
    tests
)
include(GoogleTest)
gtest_discover_tests(
    test_tree
)

# ---- CUDA
# pass options to NVCC
# set(
#     CUDA_NVCC_FLAGS
#     ${CUDA_NVCC_FLAGS};
#     -O3 -gencode arch=compute_22,code=sm_22
#     )
# specify include directories
# include_directories(
#     kernels
#     utility
#     )
# specify library paths
# link_directories(
#     /opt/foobar/lib
#     /opt/joestuff/lib
#     )
# for compilation ...
# specify target & source files to compile it from
# cuda_add_executable(
#     hellocuda
#     hellocuda.cu
#     hellocuda.h
#     kernels/hellokernels.cu
#     kernels/hellokernels.h
#     utility/wrapper.cpp
#     utility/wrapper.h
#     )

# For linking ...
# Specify target & libraries to link it with
# target_link_libraries(
#     hellocuda
#     -lfoobar
#     -ljoestuff
#     )
