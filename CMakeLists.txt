# ---- cmake for CUDA with googleTest
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(SPOCK
    DESCRIPTION "Solver for risk-averse optimal control problems."
    HOMEPAGE_URL "https://github.com/ruairimoran/raocp-parallel"
    LANGUAGES CXX)
# ----
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")  # or, wherever your cuda toolkit is stored
set(CMAKE_CUDA_ARCHITECTURES 86)
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
set(cuda_flags -arch=sm_60 -std=c++20 -Xcompiler=-Wall -Xcudafe=--display_error_number -g)
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
set(GPUTILS_BUILD_TESTING OFF)
include(FetchContent)
FetchContent_Declare(
    gputils
    URL https://github.com/GPUEngineering/GPUtils/archive/refs/heads/main.zip  # specify release version
)
FetchContent_MakeAvailable(gputils)
FetchContent_GetProperties(
    gputils
    SOURCE_DIR GPUTILS_SOURCE_DIR
)
# ----
add_subdirectory(src)
if(NOT SPOCK_BUILD_TESTING)
    set(SPOCK_BUILD_TESTING OFF)
endif()
if (SPOCK_BUILD_TESTING)
    add_subdirectory(tests)
endif()
unset(SPOCK_BUILD_TESTING CACHE)
# ----
add_executable(spock_main)
target_sources(spock_main  # add files
    PRIVATE
    main.cu
)
target_link_libraries(spock_main
    PRIVATE
    spock_compiler_flags
    spock_src
    cublas
    cusolver
    cudadevrt
)
target_include_directories(spock_main
    PRIVATE
    "${GPUTILS_SOURCE_DIR}/include"  # includes GPUtils directory
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
)
# ----
add_executable(spock_play)
target_sources(spock_play
    PRIVATE
    playground/play.cu
)
target_link_libraries(spock_play
    PRIVATE
    spock_compiler_flags
    spock_src
    cublas
    cusolver
    cudadevrt
)
target_include_directories(spock_play
    PRIVATE
    "${GPUTILS_SOURCE_DIR}/include"  # includes GPUtils directory
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
)
# ----
