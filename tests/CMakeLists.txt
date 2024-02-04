# ---- cmake with googleTest and CUDA nvcc
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SPOCK
    DESCRIPTION "Solver for risk-averse optimal control problems."
    HOMEPAGE_URL "https://github.com/ruairimoran/raocp-parallel"
    LANGUAGES CXX)
# ----
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")    
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
# ----
include_directories("home/biggirl/github/raocp-parallel")
# ----
add_library(spock_compiler_flags INTERFACE)
target_compile_features(spock_compiler_flags INTERFACE cxx_std_20)
set(CMAKE_CXX_EXTENSIONS OFF)
# ----
add_library(developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -arch=sm_60 -Xcompiler=-Wall -Xcudafe=--display_error_number)
target_compile_options(developer_flags INTERFACE
    # flags for CXX builds
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    # flags for CUDA builds
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
target_link_libraries(spock_compiler_flags INTERFACE $<BUILD_INTERFACE:developer_flags>)
# ----
add_library(spock_lib STATIC)
target_sources(spock_lib PRIVATE  # add .h files
    empty.cpp  # placeholder
)
target_sources(spock_lib PRIVATE  # add .cuh files
    ../src/tree.cuh 
    ../src/cones.cuh
)
set_target_properties(spock_lib PROPERTIES LINKER_LANGUAGE CUDA)
# ----
target_link_libraries(spock_lib PUBLIC spock_compiler_flags -lcublas)
target_include_directories(spock_lib
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    INTERFACE $<INSTALL_INTERFACE:include/spock>)
# ----
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# for Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# ----
enable_testing()
add_executable(spock_tests)
target_sources(spock_tests PRIVATE  # add .cpp files
    empty.cpp  # placeholder
)
target_sources(spock_tests PRIVATE  # add .cu files
    # test_tree.cu 
    test_cones.cu
)
target_link_libraries(spock_tests PRIVATE spock_lib GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(spock_tests)
# ----
