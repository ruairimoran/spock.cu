add_executable(networked)
target_sources(networked
    PRIVATE
    main.cu
)
target_link_libraries(networked
    PRIVATE
    spock_compiler_flags
    spock_src
    cublas
    cusolver
    cudadevrt
)
target_include_directories(networked
    PRIVATE
    "${GPUTILS_SOURCE_DIR}/include"  # includes GPUtils directory
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
)