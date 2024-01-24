# ---- cmake with googleTest and CUDA nvcc
cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
project(raocp-parallel
    DESCRIPTION "Solver for risk-averse optimal control problems."
    HOMEPAGE_URL "https://github.com/ruairimoran/raocp-parallel"
    LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# for Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
add_library(
    cuda_lib
    SHARED
    test_mat_mul.h
    test_mat_mul.cuh
)
target_compile_features(
    cuda_lib 
    PRIVATE 
    cuda_std_17)
set_target_properties(
    cuda_lib 
    PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    LINKER_LANGUAGE CUDA
    #CUDA_SEPARABLE_COMPILATION ON
)
add_executable(
    test_all
    test_tree.cc
    test_mat_mul.cc
)
target_compile_features(
    test_all
    PRIVATE 
    cxx_std_17)
#set_target_properties(
#    test_all 
#    PROPERTIES
#    CXX_STANDARD 17)
target_link_libraries(
    test_all
    GTest::gtest_main
    cuda_lib
)
include(GoogleTest)
gtest_discover_tests(
    test_all
)
