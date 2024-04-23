# ----
add_library(spock_src STATIC)
target_sources(spock_src  # add files
    PRIVATE
        definitions.cu
)
target_link_libraries(spock_src 
    PUBLIC 
        spock_compiler_flags
        cublas
        cusolver
        cudadevrt)
# ----
