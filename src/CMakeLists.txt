# ----
add_library(spock_cpu_svd STATIC cpu_svd.cpp)
target_include_directories(spock_cpu_svd PRIVATE "${eigen_SOURCE_DIR}")
target_compile_features(spock_cpu_svd PRIVATE cxx_std_20)
target_compile_definitions(spock_cpu_svd PRIVATE EIGEN_DONT_PARALLELIZE EIGEN_RUNTIME_NO_MALLOC)

add_library(spock_src STATIC)
target_sources(spock_src
    PRIVATE
    definitions.cu
    anderson.cuh
    anderson_qr.cuh
    cache.cuh
    cuda_graph.cuh
    kernel_projection.cuh
    cones.cuh
    constraints.cuh
    costs.cuh
    cost_kernels.cuh
    dynamics.cuh
    dynamics_kernels.cuh
    partial_lifting.cuh
    partial_lifting_kernels.cuh
    operator.cuh
    problem.cuh
    projections.cuh
    risks.cuh
    spock.cuh
    tree.cuh
)
target_link_libraries(spock_src
    PUBLIC
    spock_compiler_flags
    spock_cpu_svd
    cublas
    cusolver
    cudadevrt
)
target_include_directories(spock_src
    PRIVATE
    "${GPUTILS_SOURCE_DIR}/include"  # includes GPUtils directory
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
)
# ----
