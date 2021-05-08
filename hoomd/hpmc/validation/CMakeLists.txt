enable_testing()

macro(add_script_test_cpu_mpi script start stop)
    # execute on two processors
    SET(nproc 2)
    if (ENABLE_MPI)
        foreach(USER_VAR RANGE ${start} ${stop})
            add_test(NAME ${script}-mpi-cpu-${USER_VAR}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=cpu" "--user=${USER_VAR}")
                set_tests_properties(${script}-mpi-cpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
                    LABELS "validation")
        endforeach()
    endif(ENABLE_MPI)
endmacro()

macro(add_script_test_gpu_mpi script start stop)
    # execute on two processors
    SET(nproc 2)
    if (ENABLE_MPI)
        foreach(USER_VAR RANGE ${start} ${stop})
            add_test(NAME ${script}-mpi-gpu-${USER_VAR}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=gpu" "--gpu_error_checking" "--user=${USER_VAR}")
            set_tests_properties(${script}-mpi-gpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
                LABELS "validation")
        endforeach()
    endif(ENABLE_MPI)
endmacro()


macro(add_script_test_cpu script start stop)
    foreach(USER_VAR RANGE ${start} ${stop})
        add_test(${script}-cpu-${USER_VAR} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=cpu" "--user=${USER_VAR}")
        set_tests_properties(${script}-cpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
            LABELS "validation")
    endforeach()
endmacro()

macro(add_script_test_gpu script start stop)
    foreach(USER_VAR RANGE ${start} ${stop})
        add_test(${script}-gpu-${USER_VAR} ${PYTHON_EXECUTABLE} "-u" ${CMAKE_CURRENT_SOURCE_DIR}/${script} "--mode=gpu" "--user=${USER_VAR}")
        set_tests_properties(${script}-gpu-${USER_VAR} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
            LABELS "validation")
    endforeach()
endmacro()

# format: test_script user_min user_max
set(TEST_LIST_CPU
disks_implicit.py 0 0
spheres_implicit.py 0 1
spheres_implicit_repulsive.py 0 0 # no clusters
sphere_eos.py 0 0
disk_eos.py 0 0
)

set(TEST_LIST_CPU_MPI ${TEST_LIST_CPU_MPI}
disks_implicit.py 0 0
spheres_implicit.py 0 0 # no clusters
sphere_eos.py 0 0
disk_eos.py 0 0
)

if(BUILD_JIT)
set(TEST_LIST_CPU ${TEST_LIST_CPU}
    wca_dumbbell.py 0 5
    lj_spheres.py 0 0
)
set(TEST_LIST_GPU ${TEST_LIST_GPU}
    wca_dumbbell.py 0 5
    lj_spheres.py 0 0
)
set(TEST_LIST_CPU_MPI ${TEST_LIST_CPU_MPI}
    wca_dumbbell.py 0 2 # no clusters
    lj_spheres.py 0 0
)
set(TEST_LIST_GPU_MPI ${TEST_LIST_GPU_MPI}
    wca_dumbbell.py 0 2 # no clusters
    lj_spheres.py 0 0
)
endif(BUILD_JIT)

set(TEST_LIST_GPU ${TEST_LIST_GPU}
disks_implicit.py 0 0
spheres_implicit.py 0 1
sphere_eos.py 0 0
disk_eos.py 0 0
)

set(TEST_LIST_GPU_MPI ${TEST_LIST_GPU_MPI}
disks_implicit.py 0 0
spheres_implicit.py 0 0 # no clusters
sphere_eos.py 0 0
disk_eos.py 0 0
)

list(LENGTH TEST_LIST_CPU _max_i)
math(EXPR _max_i "${_max_i}-1")
foreach (CUR_I RANGE 0 ${_max_i} 3)
    math(EXPR CUR_I_START "${CUR_I}+1")
    math(EXPR CUR_I_STOP "${CUR_I}+2")
    list(GET TEST_LIST_CPU ${CUR_I} CUR_TEST)
    list(GET TEST_LIST_CPU ${CUR_I_START} CUR_START)
    list(GET TEST_LIST_CPU ${CUR_I_STOP} CUR_STOP)
    if (TEST_CPU_IN_GPU_BUILDS OR NOT ENABLE_HIP)
        add_script_test_cpu(${CUR_TEST} ${CUR_START} ${CUR_STOP})
    endif()
endforeach()

if (ENABLE_HIP)
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
endif (ENABLE_HIP)

list(LENGTH TEST_LIST_CPU_MPI _max_i)
math(EXPR _max_i "${_max_i}-1")
foreach (CUR_I RANGE 0 ${_max_i} 3)
    math(EXPR CUR_I_START "${CUR_I}+1")
    math(EXPR CUR_I_STOP "${CUR_I}+2")
    list(GET TEST_LIST_CPU_MPI ${CUR_I} CUR_TEST)
    list(GET TEST_LIST_CPU_MPI ${CUR_I_START} CUR_START)
    list(GET TEST_LIST_CPU_MPI ${CUR_I_STOP} CUR_STOP)
    if (TEST_CPU_IN_GPU_BUILDS OR NOT ENABLE_HIP)
        add_script_test_cpu_mpi(${CUR_TEST} ${CUR_START} ${CUR_STOP})
    endif()
endforeach ()

if (ENABLE_HIP)
list(LENGTH TEST_LIST_GPU_MPI _max_i)
math(EXPR _max_i "${_max_i}-1")
foreach (CUR_I RANGE 0 ${_max_i} 3)
    math(EXPR CUR_I_START "${CUR_I}+1")
    math(EXPR CUR_I_STOP "${CUR_I}+2")
    list(GET TEST_LIST_GPU_MPI ${CUR_I} CUR_TEST)
    list(GET TEST_LIST_GPU_MPI ${CUR_I_START} CUR_START)
    list(GET TEST_LIST_GPU_MPI ${CUR_I_STOP} CUR_STOP)
    add_script_test_gpu_mpi(${CUR_TEST} ${CUR_START} ${CUR_STOP})
endforeach()
endif(ENABLE_HIP)
