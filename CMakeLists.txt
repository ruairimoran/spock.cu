# ---- cmake with CUDA nvcc
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
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
target_compile_features(spock_compiler_flags INTERFACE cxx_std_11)
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
        tests/empty.cpp  # placeholder
)
target_sources(spock_lib PRIVATE  # add .cuh files
        src/tree.cuh
)
set_target_properties(spock_lib PROPERTIES LINKER_LANGUAGE CUDA)
# ----
target_link_libraries(spock_lib PUBLIC spock_compiler_flags)
target_include_directories(spock_lib
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    INTERFACE $<INSTALL_INTERFACE:include/spock>)
# ----
add_executable(spock)
target_sources(spock PRIVATE  # add .cpp files
        tests/empty.cpp  # placeholder
)
target_sources(spock PRIVATE  # add .cu files
        tests/empty.cpp  # placeholder
)
target_link_libraries(spock PRIVATE spock_lib)
# ----
