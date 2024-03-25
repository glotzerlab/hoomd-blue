###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
    test_cell_list
    test_cell_list_stencil
    test_gpu_array
    test_global_array
    test_gridshift_correct
    test_index1d
    test_messenger
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

if (HIP_PLATFORM STREQUAL "nvcc")
    set(add TEST_LIST test_gpu_polymorph)
endif()

if (ENABLE_HIP)
list(APPEND TEST_LIST
     test_warp_tools
     )
set(SYNC_TEST_LIST
    test_warp_tools)
endif (ENABLE_HIP)

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
    if(ENABLE_HIP AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
        set(_cuda_sources ${CUR_TEST}.cu)
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    else()
        set(_cuda_sources "")
    endif()

    add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cc ${_cuda_sources})
    if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set_source_files_properties(${CUR_TEST}.cc PROPERTIES COMPILE_FLAGS "-Wno-self-assign-overloaded")
    endif()

    add_dependencies(test_all ${CUR_TEST})
    target_link_libraries(${CUR_TEST} _hoomd pybind11::embed)

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
if(ENABLE_HIP AND HIP_PLATFORM STREQUAL "nvcc")

foreach (CUR_TEST ${SYNC_TEST_LIST})
    if (ENABLE_MPI)
        add_test(NAME ${CUR_TEST}-synccheck
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
                         ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool synccheck $<TARGET_FILE:${CUR_TEST}>)
    else()
        add_test(NAME ${CUR_TEST}-synccheck COMMAND ${CUDA_MEMCHECK_EXECUTABLE} --error-exitcode 123 --tool synccheck $<TARGET_FILE:${CUR_TEST}>)
    endif()
endforeach (CUR_TEST)
endif()
