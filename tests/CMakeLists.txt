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
target_sources(spock_tests  # add files
    PRIVATE
        test_tree.cu 
        test_problem.cu
        test_cones.cu
)
target_link_libraries(spock_tests 
    PRIVATE 
        spock_src
        GTest::gtest 
        GTest::gtest_main)
target_include_directories(spock_tests
    PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/src"
)
include(GoogleTest)
gtest_discover_tests(spock_tests
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
# ----
