# Maintainer: joaander

###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
    test_cell_list
    test_cell_list_stencil
    test_gpu_array
    test_global_array
    test_gpu_polymorph
    test_gridshift_correct
    test_index1d
    test_messenger
    test_particle_group
    test_pdata
    test_quat
    test_rotmat2
    test_rotmat3
    test_shared_signal
    test_system
    test_utils
    test_vec2
    test_vec3
    random_numbers_test
    )

if (ENABLE_CUDA)
list(APPEND TEST_LIST
     test_warp_tools
     )
set(SYNC_TEST_LIST
    test_warp_tools)
endif (ENABLE_CUDA)

option(HOOMD_SKIP_LONG_TESTS "Skip long unit tests" on)

if(ENABLE_MPI)
    MACRO(ADD_TO_MPI_TESTS _KEY _VALUE)
    SET("NProc_${_KEY}" "${_VALUE}")
    SET(MPI_TEST_LIST ${MPI_TEST_LIST} ${_KEY})
    ENDMACRO(ADD_TO_MPI_TESTS)

    # define every test together with the number of processors
    ADD_TO_MPI_TESTS(test_load_balancer 8)
endif()

foreach (CUR_TEST ${TEST_LIST} ${MPI_TEST_LIST})
    # add and link the unit test executable
    if(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
        CUDA_COMPILE(_CUDA_GENERATED_FILES ${CUR_TEST}.cu OPTIONS ${CUDA_ADDITIONAL_OPTIONS})
    else()
        set(_CUDA_GENERATED_FILES "")
    endif()

    add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cc ${_CUDA_GENERATED_FILES})

    add_dependencies(test_all ${CUR_TEST})

    target_link_libraries(${CUR_TEST} _hoomd ${PYTHON_LIBRARIES} ${HOOMD_COMMON_LIBS})
    fix_cudart_rpath(${CUR_TEST})

    if (ENABLE_MPI)
        # set appropriate compiler/linker flags
        if(MPI_COMPILE_FLAGS)
            set_target_properties(${CUR_TEST} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
        endif(MPI_COMPILE_FLAGS)
        if(MPI_LINK_FLAGS)
            set_target_properties(${CUR_TEST} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        endif(MPI_LINK_FLAGS)
    endif (ENABLE_MPI)
endforeach (CUR_TEST)

# add non-MPI tests to test list first
foreach (CUR_TEST ${TEST_LIST})
    # add it to the unit test list
    if (ENABLE_MPI)
        add_test(NAME ${CUR_TEST} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} $<TARGET_FILE:${CUR_TEST}>)
    else()
        add_test(NAME ${CUR_TEST} COMMAND $<TARGET_FILE:${CUR_TEST}>)
    endif()
endforeach(CUR_TEST)

# add MPI tests
foreach (CUR_TEST ${MPI_TEST_LIST})
    # add it to the unit test list
    # add mpi- prefix to distinguish these tests
    set(MPI_TEST_NAME mpi-${CUR_TEST})

    add_test(NAME ${MPI_TEST_NAME} COMMAND
             ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
             ${NProc_${CUR_TEST}} ${MPIEXEC_POSTFLAGS}
             $<TARGET_FILE:${CUR_TEST}>)
endforeach(CUR_TEST)

# run tests under cuda-memcheck --tool synccheck
if(ENABLE_CUDA)
# cuda-memcheck executable
find_program(CUDA_MEMCHECK_EXECUTABLE
  NAMES cuda-memcheck
  PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin"
        "${CUDA_TOOLKIT_ROOT_DIR}/bin64"
  ENV CUDA_BIN_PATH
  NO_DEFAULT_PATH
  )
mark_as_advanced(CUDA_MEMCHECK_EXECUTABLE)

foreach (CUR_TEST ${SYNC_TEST_LIST})
    if (ENABLE_MPI)
        add_test(NAME ${CUR_TEST}-synccheck
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                         ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool synccheck $<TARGET_FILE:${CUR_TEST}>)
    else()
        add_test(NAME ${CUR_TEST}-synccheck COMMAND ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool synccheck $<TARGET_FILE:${CUR_TEST}>)
    endif()
endforeach (CUR_TEST)
endif(ENABLE_CUDA)
