enable_testing()

macro(add_script_test_cpu_mpi script USER_VAR)
    # execute on two processors
    SET(nproc 2)
    if (ENABLE_MPI)
        if(NOT "${EXCLUDE_FROM_MPI}" MATCHES ${script})
            add_test(NAME ${script}-mpi-cpu-${USER_VAR}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=cpu" "--user=${USER_VAR}")
                set_tests_properties(${script}-mpi-cpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
                    LABELS "validation")
            endif()
    endif(ENABLE_MPI)
endmacro()

macro(add_script_test_gpu_mpi script USER_VAR)
    # execute on two processors
    SET(nproc 2)
    if (ENABLE_MPI)
        if(NOT "${EXCLUDE_FROM_GPU_MPI}" MATCHES ${script})
            add_test(NAME ${script}-mpi-gpu-${USER_VAR}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=gpu" "--gpu_error_checking" "--user=${USER_VAR}")
            set_tests_properties(${script}-mpi-gpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
                LABELS "validation")
        endif()
    endif(ENABLE_MPI)
endmacro()


macro(add_script_test_cpu script start stop)
    foreach(USER_VAR RANGE ${start} ${stop})
        add_test(${script}-cpu-${USER_VAR} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=cpu" "--user=${USER_VAR}")
        set_tests_properties(${script}-cpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
            LABELS "validation")
        add_script_test_cpu_mpi(${script} ${USER_VAR})
    endforeach()
endmacro()

macro(add_script_test_gpu script start stop)
    foreach(USER_VAR RANGE ${start} ${stop})
        add_test(${script}-gpu-${USER_VAR} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=gpu" "--user=${USER_VAR}")
        set_tests_properties(${script}-gpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
            LABELS "validation")
        add_script_test_gpu_mpi(${script} ${USER_VAR})
    endforeach()
endmacro()

# format: test_script user_min user_max
set(TEST_LIST_CPU
minimize_fire_rigid.py 0 0
bd_angular.py 0 0
compare_npt_nvt_rigid.py 0 2
npt_dimer_eos.py 0 2
)

set(TEST_LIST_GPU
minimize_fire_rigid.py 0 0
bd_angular.py 0 0
compare_npt_nvt_rigid.py 0 2
npt_dimer_eos.py 0 2
)

set(EXCLUDE_FROM_GPU_MPI
bd_angular.py
)

set(EXCLUDE_FROM_MPI
bd_angular.py
)

list(LENGTH TEST_LIST_CPU _max_i)
math(EXPR _max_i "${_max_i}-1")
foreach (CUR_I RANGE 0 ${_max_i} 3)
    math(EXPR CUR_I_START "${CUR_I}+1")
    math(EXPR CUR_I_STOP "${CUR_I}+2")
    list(GET TEST_LIST_CPU ${CUR_I} CUR_TEST)
    list(GET TEST_LIST_CPU ${CUR_I_START} CUR_START)
    list(GET TEST_LIST_CPU ${CUR_I_STOP} CUR_STOP)
    if (TEST_CPU_IN_GPU_BUILDS OR NOT ENABLE_CUDA)
        add_script_test_cpu(${CUR_TEST} ${CUR_START} ${CUR_STOP})
    endif()
endforeach ()

if (ENABLE_CUDA)
list(LENGTH TEST_LIST_GPU _max_i)
math(EXPR _max_i "${_max_i}-1")
foreach (CUR_I RANGE 0 ${_max_i} 3)
    math(EXPR CUR_I_START "${CUR_I}+1")
    math(EXPR CUR_I_STOP "${CUR_I}+2")
    list(GET TEST_LIST_GPU ${CUR_I} CUR_TEST)
    list(GET TEST_LIST_GPU ${CUR_I_START} CUR_START)
    list(GET TEST_LIST_GPU ${CUR_I_STOP} CUR_STOP)
    add_script_test_gpu(${CUR_TEST} ${CUR_START} ${CUR_STOP})
endforeach ()
endif (ENABLE_CUDA)
