function(add_spock_mpc_example target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE spock_compiler_flags spock_src cublas cusolver cudadevrt)
    target_include_directories(${target} PRIVATE
        "${GPUTILS_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/include")
endfunction()

add_spock_mpc_example(networked_mpc main.cu)
add_spock_mpc_example(networked_mpc_persistent persistent.cu)
add_spock_mpc_example(networked_mpc_persistent_aa persistent.cu)
add_spock_mpc_example(networked_mpc_persistent_qr persistent.cu)
add_spock_mpc_example(check_anderson check_anderson.cu)
add_spock_mpc_example(check_anderson_qr check_anderson_qr.cu)
add_spock_mpc_example(check_qr_workspace check_qr_workspace.cu)
add_spock_mpc_example(benchmark_dynamics benchmark_dynamics.cu)

if(SPOCK_BUILD_TEST)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(validation_dir "${CMAKE_CURRENT_BINARY_DIR}/test-results")
    file(MAKE_DIRECTORY "${validation_dir}")
    function(add_anderson_validation name script)
        add_test(NAME "${name}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}"
                ${ARGN} --report "${validation_dir}/${name}.json")
        set_tests_properties("${name}" PROPERTIES RESOURCE_LOCK spock_gpu
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    endfunction()
    foreach(precision IN ITEMS double float)
        set(rolling_flags)
        if(precision STREQUAL "float")
            list(APPEND rolling_flags --float)
        endif()
        foreach(mode IN ITEMS full update)
            set(mode_flags)
            if(mode STREQUAL "update")
                list(APPEND mode_flags --qr-updates)
            endif()
            add_anderson_validation("spock_anderson_${mode}_${precision}" test_anderson_reference.py
                --gpu-executable "$<TARGET_FILE:check_anderson>"
                --gpu-precision "${precision}" --include-nonfinite ${mode_flags})
            add_anderson_validation("spock_anderson_cpu_${mode}_${precision}" test_anderson_reference.py
                --gpu-executable "$<TARGET_FILE:check_anderson>" --fit-backend cpu
                --gpu-precision "${precision}" --include-nonfinite ${mode_flags})
        endforeach()
        add_anderson_validation("spock_rolling_qr_${precision}" test_anderson_qr.py
            --executable "$<TARGET_FILE:check_anderson_qr>" ${rolling_flags})
    endforeach()
    add_test(NAME spock_qr_workspace COMMAND "$<TARGET_FILE:check_qr_workspace>")
    set_tests_properties(spock_qr_workspace PROPERTIES RESOURCE_LOCK spock_gpu)
    add_test(NAME spock_worker_responses
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_worker_responses.py")
    add_test(NAME spock_online_model
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_online_model.py" -v)
endif()
