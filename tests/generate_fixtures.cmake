# Keep generated tensors in the build tree. Both precisions are needed by
# each CUDA test; generate them sequentially because they share metadata.
file(MAKE_DIRECTORY "${FIXTURE_DIR}")
foreach(precision IN ITEMS f d)
    execute_process(
        COMMAND "${PYTHON}" "${GENERATOR}" --dt "${precision}"
        WORKING_DIRECTORY "${FIXTURE_DIR}"
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Failed to generate ${precision}-precision SPOCK fixtures")
    endif()
endforeach()
execute_process(
    COMMAND "${PYTHON}" "${DENSE_GENERATOR}"
    WORKING_DIRECTORY "${FIXTURE_DIR}"
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Failed to generate dense-cost fixtures")
endif()
execute_process(
    COMMAND "${PYTHON}" "${DIAGONAL_GENERATOR}"
    WORKING_DIRECTORY "${FIXTURE_DIR}"
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Failed to generate diagonal-cost fixtures")
endif()
execute_process(
    COMMAND "${PYTHON}" "${DYNAMICS_GENERATOR}"
    WORKING_DIRECTORY "${FIXTURE_DIR}"
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Failed to generate dynamics-projection fixtures")
endif()
