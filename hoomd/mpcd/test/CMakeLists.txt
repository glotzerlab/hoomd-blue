# Maintainer: mphoward

###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
    )

if(ENABLE_MPI)
    MACRO(ADD_TO_MPI_TESTS _KEY _VALUE)
    SET("NProc_${_KEY}" "${_VALUE}")
    SET(MPI_TEST_LIST ${MPI_TEST_LIST} ${_KEY})
    ENDMACRO(ADD_TO_MPI_TESTS)

    # define every test together with the number of processors
    ADD_TO_MPI_TESTS(communicator 8)
endif()

foreach (CUR_TEST ${TEST_LIST} ${MPI_TEST_LIST})
    set(TEST_NAME ${CUR_TEST}_test)
    # Need to define NO_IMPORT_ARRAY in every file but hoomd_module.cc
    set_source_files_properties(${TEST_NAME}.cc PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

    # add and link the unit test executable
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_NAME}.cc)

    add_dependencies(test_all ${TEST_NAME})

    target_link_libraries(${TEST_NAME} _hoomd _md _mpcd ${HOOMD_COMMON_LIBS})
    fix_cudart_rpath(${TEST_NAME})

    if (ENABLE_MPI)
        # set appropriate compiler/linker flags
        if(MPI_COMPILE_FLAGS)
            set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
        endif(MPI_COMPILE_FLAGS)
        if(MPI_LINK_FLAGS)
            set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        endif(MPI_LINK_FLAGS)
    endif (ENABLE_MPI)
endforeach (CUR_TEST)

# add non-MPI tests to test list first
foreach (CUR_TEST ${TEST_LIST})
    # add it to the unit test list
    get_target_property(CUR_TEST_EXE ${CUR_TEST} LOCATION)

    if (ENABLE_MPI)
        add_test(${CUR_TEST} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} ${CUR_TEST_EXE})
    else()
        add_test(${CUR_TEST} ${CUR_TEST_EXE})
    endif()
endforeach(CUR_TEST)

# add MPI tests
foreach (CUR_TEST ${MPI_TEST_LIST})
    # add it to the unit test list
    get_target_property(CUR_TEST_EXE ${CUR_TEST}_test LOCATION)

    # add mpi- prefix to distinguish these tests
    set(MPI_TEST_NAME mpcd-${CUR_TEST}-mpi)

    add_test(${MPI_TEST_NAME}
             ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
             ${NProc_${CUR_TEST}} ${MPIEXEC_POSTFLAGS}
             ${CUR_TEST_EXE})
endforeach(CUR_TEST)
