###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
    test_aabb_tree
    test_convex_polygon
    test_convex_polyhedron
    test_ellipsoid
    test_faceted_sphere
    test_moves
    test_polyhedron
    test_simple_polygon
    test_sphere
    test_sphere_union
    test_spheropolygon
    test_spheropolyhedron
    test_sphinx
    )

foreach (CUR_TEST ${TEST_LIST})
    # add and link the unit test executable
    if(ENABLE_HIP AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
        set(_cuda_sources ${CUR_TEST}.cu)
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    else()
        set(_cuda_sources "")
    endif()

    add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cc ${_cuda_sources})

    add_dependencies(test_all ${CUR_TEST})

    target_link_libraries(${CUR_TEST} _hpmc pybind11::embed)

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
