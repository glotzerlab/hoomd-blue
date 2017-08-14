# Maintainer: joaander

#############################
# macro for adding hoomd script tests
macro(add_hoomd_script_test test_py)
    # name the test
    get_filename_component(_test_name ${test_py} NAME_WE)

    # use mpirun -n 1 in MPI builds, otherwise, just run hoomd
    if (ENABLE_MPI)
        add_test(NAME script-${_test_name}-cpu
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                ${PYTHON_EXECUTABLE} ${test_py} "--mode=cpu" "--gpu_error_checking")
        set_tests_properties(script-${_test_name}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
        if (ENABLE_CUDA)
            add_test(NAME script-${_test_name}-gpu
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                    ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu" "--gpu_error_checking")
            set_tests_properties(script-${_test_name}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
        endif (ENABLE_CUDA)
    else ()
        add_test(NAME script-${_test_name}-cpu COMMAND ${PYTHON_EXECUTABLE} ${test_py} "--mode=cpu" "--gpu_error_checking")
        set_tests_properties(script-${_test_name}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
        if (ENABLE_CUDA)
            add_test(NAME script-${_test_name}-gpu COMMAND ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu" "--gpu_error_checking")
            set_tests_properties(script-${_test_name}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
        endif (ENABLE_CUDA)
    endif()

endmacro(add_hoomd_script_test)
###############################

# loop through all test_*.py files
file(GLOB _hoomd_script_tests ${CMAKE_CURRENT_SOURCE_DIR}/test_*.py)

foreach(test ${_hoomd_script_tests})
    add_hoomd_script_test(${test})
endforeach(test)