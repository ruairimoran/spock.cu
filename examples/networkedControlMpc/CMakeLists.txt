add_executable(networked_mpc main.cu)
target_link_libraries(networked_mpc PRIVATE spock_compiler_flags spock_src cublas cusolver cudadevrt)
target_include_directories(networked_mpc PRIVATE
    "${GPUTILS_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include")
