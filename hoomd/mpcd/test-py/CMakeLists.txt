# Maintainer: mphoward

set(TEST_LIST
    collide_at
    collide_srd
    data_snapshot
    data_system
    force_block
    force_constant
    force_sine
    init_make_random
    integrate_integrator
    integrate_slit
    integrate_slit_pore
    stream_bulk
    stream_slit
    stream_slit_pore
    update_sort
    )
SET(EXCLUDE_FROM_MPI
    )

#############################
# macro for adding hoomd script tests
macro(add_hoomd_script_test test)

set(TEST_NAME mpcd-${test})
set(TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${test}_test.py)

# use mpirun -n 1 in MPI builds, otherwise, just run hoomd
if (ENABLE_MPI)
    add_test(NAME ${TEST_NAME}-cpu
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                     ${PYTHON_EXECUTABLE} ${TEST_FILE} "--mode=cpu" "--gpu_error_checking")
    set_tests_properties(${TEST_NAME}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    if (ENABLE_CUDA)
        add_test(NAME ${TEST_NAME}-gpu
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                         ${PYTHON_EXECUTABLE} ${TEST_FILE} "--mode=gpu" "--gpu_error_checking")
    set_tests_properties(${TEST_NAME}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif (ENABLE_CUDA)
else ()
    add_test(NAME ${TEST_NAME}-cpu COMMAND ${PYTHON_EXECUTABLE} ${TEST_FILE} "--mode=cpu" "--gpu_error_checking")
    set_tests_properties(${TEST_NAME}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    if (ENABLE_CUDA)
        add_test(NAME ${TEST_NAME}-gpu COMMAND ${PYTHON_EXECUTABLE} ${TEST_FILE} "--mode=gpu" "--gpu_error_checking")
        set_tests_properties(${TEST_NAME}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif (ENABLE_CUDA)
endif()

endmacro(add_hoomd_script_test)
###############################

#############################
# macro for adding hoomd script tests (MPI version)
macro(add_hoomd_script_test_mpi test nproc)

set(TEST_NAME mpcd-${test})
set(TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${test}_test.py)

add_test(NAME ${TEST_NAME}-mpi-cpu
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
         ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} ${TEST_FILE} "--mode=cpu" "--gpu_error_checking")
set_tests_properties(${TEST_NAME}-mpi-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
if (ENABLE_CUDA)
add_test(NAME ${TEST_NAME}-mpi-gpu
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
         ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} ${TEST_FILE} "--mode=gpu" "--gpu_error_checking")
set_tests_properties(${TEST_NAME}-mpi-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
endif (ENABLE_CUDA)
endmacro(add_hoomd_script_test_mpi)
###############################

# setup regular tests
foreach(test ${TEST_LIST})
add_hoomd_script_test(${test})
endforeach(test)

# setup MPI tests
if (ENABLE_MPI)
    foreach(test ${TEST_LIST})
        GET_FILENAME_COMPONENT(test_name ${test} NAME_WE)
        if(NOT "${EXCLUDE_FROM_MPI}" MATCHES ${test_name})
            # execute on two processors
            add_hoomd_script_test_mpi(${test} 2)
        endif()
    endforeach(test)
endif(ENABLE_MPI)
