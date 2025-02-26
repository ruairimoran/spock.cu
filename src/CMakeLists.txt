# ----
add_library(spock_src STATIC)
target_sources(spock_src
    PRIVATE
    definitions.cu
    cache.cuh
    cones.cuh
    constraints.cuh
    costs.cuh
    dynamics.cuh
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
