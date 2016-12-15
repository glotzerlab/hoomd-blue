# Maintainer: joaander

#############################
# macro for adding hoomd script tests
macro(add_hoomd_script_test test_py)
# name the test
get_filename_component(_test_name ${test_py} NAME_WE)

# use mpirun -n 1 in MPI builds, otherwise, just run hoomd
if (ENABLE_MPI)
    if (TEST_CPU_IN_GPU_BUILDS OR NOT ENABLE_CUDA)
        add_test(NAME script-${_test_name}-cpu
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                         ${PYTHON_EXECUTABLE} ${test_py} "--mode=cpu" "--gpu_error_checking")
        set_tests_properties(script-${_test_name}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif()

    if (ENABLE_CUDA)
        add_test(NAME script-${_test_name}-gpu
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                         ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu" "--gpu_error_checking")
    set_tests_properties(script-${_test_name}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif (ENABLE_CUDA)
else ()
    if (TEST_CPU_IN_GPU_BUILDS OR NOT ENABLE_CUDA)
        add_test(NAME script-${_test_name}-cpu COMMAND ${PYTHON_EXECUTABLE} ${test_py} "--mode=cpu" "--gpu_error_checking")
        set_tests_properties(script-${_test_name}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif()

    if (ENABLE_CUDA)
        add_test(NAME script-${_test_name}-gpu COMMAND ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu" "--gpu_error_checking")
        set_tests_properties(script-${_test_name}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif (ENABLE_CUDA)
endif()

endmacro(add_hoomd_script_test)
###############################

#############################
# macro for adding hoomd script tests (MPI version)
macro(add_hoomd_script_test_mpi test_py nproc)
# name the test
get_filename_component(_test_name ${test_py} NAME_WE)

if (TEST_CPU_IN_GPU_BUILDS OR NOT ENABLE_CUDA)
    add_test(NAME script-${_test_name}-mpi-cpu
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
             ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} ${test_py} "--mode=cpu" "--gpu_error_checking")
    set_tests_properties(script-${_test_name}-mpi-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
endif()

if (ENABLE_CUDA)
    add_test(NAME script-${_test_name}-mpi-gpu
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
             ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu" "--gpu_error_checking")
    set_tests_properties(script-${_test_name}-mpi-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
endif (ENABLE_CUDA)
endmacro(add_hoomd_script_test_mpi)
###############################

#############################
# macro for adding hoomd script tests (with cuda-memcheck)
if(ENABLE_CUDA)

# cuda-memcheck executable
find_program(CUDA_MEMCHECK_EXECUTABLE
  NAMES cuda-memcheck
  PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin"
        "${CUDA_TOOLKIT_ROOT_DIR}/bin64"
  ENV CUDA_BIN_PATH
  NO_DEFAULT_PATH
  )

macro(add_hoomd_script_test_cuda_memcheck test_py)
# name the test
get_filename_component(_test_name ${test_py} NAME_WE)

if (ENABLE_CUDA)
if (ENABLE_MPI)
    add_test(NAME script-${_test_name}-racecheck-gpu
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                     ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool racecheck ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu")
    set_tests_properties(script-${_test_name}-racecheck-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    add_test(NAME script-${_test_name}-memcheck-gpu
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                     ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool memcheck ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu")
    set_tests_properties(script-${_test_name}-memcheck-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
else()
    add_test(NAME script-${_test_name}-racecheck-gpu COMMAND ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool racecheck ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu")
    set_tests_properties(script-${_test_name}-racecheck-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    add_test(NAME script-${_test_name}-memcheck-gpu COMMAND ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool memcheck ${PYTHON_EXECUTABLE} ${test_py} "--mode=gpu")
    set_tests_properties(script-${_test_name}-memcheck-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")

endif()
endif (ENABLE_CUDA)

endmacro(add_hoomd_script_test_cuda_memcheck test_py)
endif(ENABLE_CUDA)
###############################

# loop through all test_*.py files
file(GLOB _hoomd_script_tests ${CMAKE_CURRENT_SOURCE_DIR}/test_*.py)

foreach(test ${_hoomd_script_tests})
add_hoomd_script_test(${test})
endforeach(test)

# exclude some tests from MPI
SET(EXCLUDE_FROM_MPI
    )

if (ENABLE_MPI)
    foreach(test ${_hoomd_script_tests})
        GET_FILENAME_COMPONENT(test_name ${test} NAME_WE)
        if(NOT "${EXCLUDE_FROM_MPI}" MATCHES ${test_name})
            # execute on two processors
            add_hoomd_script_test_mpi(${test} 2)
        endif()
    endforeach(test)

endif(ENABLE_MPI)
