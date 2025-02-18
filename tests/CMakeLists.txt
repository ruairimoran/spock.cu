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
    cuda/testKernels.cu
    cuda/testTree.cu
    cuda/testCones.cu
    cuda/testRisks.cu
    cuda/testProblem.cu
    cuda/testProjections.cu
    cuda/testCache.cu
    cuda/testOperator.cu
)
target_link_libraries(spock_tests
    PRIVATE
    spock_src
    GTest::gtest
    GTest::gtest_main)
target_include_directories(spock_tests
    PRIVATE
    "${GPUTILS_SOURCE_DIR}/include"  # includes GPUtils directory
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
)
include(GoogleTest)
gtest_discover_tests(spock_tests)
# ----
