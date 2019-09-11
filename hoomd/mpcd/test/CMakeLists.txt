# Maintainer: mphoward

###################################
## Setup all of the test executables in a for loop
if (BUILD_TESTING)

# external_field test will not link properly due to separable compilation, disabling.
set(TEST_LIST
    at_collision_method
    cell_list
    cell_thermo_compute
    #external_field
    slit_geometry_filler
    slit_pore_geometry_filler
    sorter
    srd_collision_method
    streaming_method
    virtual_particle
    )
endif()

if(ENABLE_MPI)
    MACRO(ADD_TO_MPI_TESTS _KEY _VALUE)
    SET("NProc_${_KEY}" "${_VALUE}")
    SET(MPI_TEST_LIST ${MPI_TEST_LIST} ${_KEY})
    ENDMACRO(ADD_TO_MPI_TESTS)

    # define every test together with the number of processors
    # these are long running tests
    if (BUILD_VALIDATION)
    ADD_TO_MPI_TESTS(communicator 8)
    endif()
    # these are tests that are shorter running
    if (BUILD_TESTING)
    ADD_TO_MPI_TESTS(cell_communicator 8)
    ADD_TO_MPI_TESTS(cell_list 8)
    ADD_TO_MPI_TESTS(cell_thermo_compute 8)
    ADD_TO_MPI_TESTS(slit_geometry_filler 8)
    ADD_TO_MPI_TESTS(slit_pore_geometry_filler 8)
    endif()
endif()

macro(compile_test TEST_EXE TEST_SRC)
    # check for extra cuda files
    if(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_EXE}.cu)
        set(CUDA_SEPARABLE_COMPILATION OFF)
        CUDA_COMPILE(_CUDA_GENERATED_FILES ${TEST_EXE}.cu OPTIONS ${CUDA_ADDITIONAL_OPTIONS})
    else()
        set(_CUDA_GENERATED_FILES "")
    endif()

    # add and link the unit test executable
    add_executable(${TEST_EXE} EXCLUDE_FROM_ALL ${TEST_SRC} ${_CUDA_GENERATED_FILES})
    add_dependencies(test_all ${TEST_EXE})
    target_link_libraries(${TEST_EXE} _mpcd _md ${HOOMD_LIBRARIES} ${PYTHON_LIBRARIES})
    fix_cudart_rpath(${TEST_EXE})

    if (ENABLE_MPI)
        # set appropriate compiler/linker flags
        if(MPI_COMPILE_FLAGS)
            set_target_properties(${TEST_EXE} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
        endif(MPI_COMPILE_FLAGS)
        if(MPI_LINK_FLAGS)
            set_target_properties(${TEST_EXE} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        endif(MPI_LINK_FLAGS)
    endif (ENABLE_MPI)
endmacro(compile_test)

# add non-MPI tests to test list first
foreach (CUR_TEST ${TEST_LIST})
    set(CUR_TEST_SRC ${CUR_TEST}_test.cc)
    set(CUR_TEST_EXE ${CUR_TEST}_test)
    set(CUR_TEST_NAME mpcd-core-${CUR_TEST})

    compile_test(${CUR_TEST_EXE} ${CUR_TEST_SRC})
    if (ENABLE_MPI)
        add_test(NAME ${CUR_TEST_NAME} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} $<TARGET_FILE:${CUR_TEST_EXE}>)
    else()
        add_test(NAME ${CUR_TEST_NAME} COMMAND $<TARGET_FILE:${CUR_TEST_EXE}>)
    endif()
endforeach(CUR_TEST)

# add MPI tests
foreach (CUR_TEST ${MPI_TEST_LIST})
    set(CUR_TEST_SRC ${CUR_TEST}_mpi_test.cc)
    set(CUR_TEST_EXE ${CUR_TEST}_mpi_test)
    set(CUR_TEST_NAME mpcd-core-${CUR_TEST}-mpi)

    compile_test(${CUR_TEST_EXE} ${CUR_TEST_SRC})
    add_test(NAME ${CUR_TEST_NAME} COMMAND
             ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
             ${NProc_${CUR_TEST}} ${MPIEXEC_POSTFLAGS}
             $<TARGET_FILE:${CUR_TEST_EXE}>)
endforeach(CUR_TEST)
