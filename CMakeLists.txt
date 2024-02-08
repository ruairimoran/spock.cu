# ---- cmake for CUDA with googleTest
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(SPOCK
    DESCRIPTION "Solver for risk-averse optimal control problems."
    HOMEPAGE_URL "https://github.com/ruairimoran/raocp-parallel"
    LANGUAGES CXX)
# ----
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")    
set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)
# ----
include_directories("home/biggirl/github/raocp-parallel")
add_subdirectory(src)
add_subdirectory(tests)
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
add_executable(spock)
target_sources(spock PRIVATE  # add files
    main.cu
)
find_library(SPOCK_SRC_LOC spock_src REQUIRED ON)
target_link_libraries(spock PRIVATE 
    ${SPOCK_SRC_LOC})
# ----
