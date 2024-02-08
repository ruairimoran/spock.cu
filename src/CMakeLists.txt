# ----
add_library(spock_src STATIC)
target_sources(spock_src PRIVATE  # add files
    definitions.cu
    tree.cuh 
    problem.cuh
    cones.cuh
)
set_target_properties(spock_src PROPERTIES 
    LINKER_LANGUAGE CUDA
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(spock_src PUBLIC 
    spock_compiler_flags
    -lcublas
    -lcudadevrt)
target_include_directories(spock_src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    INTERFACE $<INSTALL_INTERFACE:include/spock>)
# ----
