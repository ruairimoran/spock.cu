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
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(SPOCK_TEST_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
add_test(NAME spock_generate_fixtures
    COMMAND "${CMAKE_COMMAND}"
        "-DPYTHON=${Python3_EXECUTABLE}"
        "-DGENERATOR=${CMAKE_CURRENT_SOURCE_DIR}/cuda/test.py"
        "-DDENSE_GENERATOR=${CMAKE_CURRENT_SOURCE_DIR}/cuda/generate_dense_cost.py"
        "-DDIAGONAL_GENERATOR=${CMAKE_CURRENT_SOURCE_DIR}/cuda/generate_diagonal_cost.py"
        "-DDYNAMICS_GENERATOR=${CMAKE_CURRENT_SOURCE_DIR}/cuda/generate_dynamics_projection.py"
        "-DFIXTURE_DIR=${SPOCK_TEST_FIXTURE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixtures.cmake")
set_tests_properties(spock_generate_fixtures PROPERTIES
    FIXTURES_SETUP spock_data
    ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/api/python:${GPUTILS_SOURCE_DIR}/python")
add_executable(spock_tests)
target_sources(spock_tests  # add files
    PRIVATE
    cuda/testKernels.cu
    cuda/testTree.cu
    cuda/testCones.cu
    cuda/testRisks.cu
    cuda/testProblem.cu
    cuda/testProjections.cu
    cuda/testOperator.cu
    cuda/testCache.cu
    cuda/testMetricProduct.cu
    cuda/testCostAliasing.cu
    cuda/testDiagonalCost.cu
    cuda/testDynamics.cu
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
target_compile_definitions(spock_tests PRIVATE
    SPOCK_TEST_DATA_DIR="${SPOCK_TEST_FIXTURE_DIR}/data/"
    SPOCK_TEST_FIXTURE_DIR="${SPOCK_TEST_FIXTURE_DIR}/")
include(GoogleTest)
gtest_discover_tests(spock_tests PROPERTIES FIXTURES_REQUIRED spock_data RESOURCE_LOCK spock_gpu)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-fixtures")
add_test(NAME spock_python
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/python" -p "test*.py" -v)
set_tests_properties(spock_python PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-fixtures"
    ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/api/python:${GPUTILS_SOURCE_DIR}/python;OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;MPLBACKEND=Agg")
# ----
