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
target_sources(spock_tests PRIVATE  # add files
    test_tree.cu 
    test_problem.cu
    test_cones.cu
)
# find_library(SPOCK_SRC_LOC spock_src.a PATHS ${CMAKE_CURRENT_SOURCE_DIR}/build REQUIRED ON)
target_link_libraries(spock_tests PRIVATE 
    ${SPOCK_SRC_LOC}
    GTest::gtest 
    GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(spock_tests)
# ----
