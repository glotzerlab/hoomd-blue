# Maintainer: joaander

###############################
# need a few settings for the unit test library
# first, they need the boost unit test framework added as a library
set(ADDITIONAL_LIBS _hoomd
        _hpmc
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        )

# boost 1.60.0 requires the boost timer and chrono be linked to unit tests
if (Boost_MINOR_VERSION GREATER 59)
list(APPEND ADDITIONAL_LIBS ${Boost_TIMER_LIBRARY} ${Boost_CHRONO_LIBRARY} ${Boost_SYSTEM_LIBRARY})
endif ()

add_definitions(-DBOOST_TEST_DYN_LINK)

####################################
## Setup valgrind options
if (UNIX AND NOT APPLE)
find_program(VALGRIND valgrind DOC
    "valgrind executable for advanced regression tests")
endif (UNIX AND NOT APPLE)

## User gets an option to enable valgrind tests
if (VALGRIND)
    option(ENABLE_VALGRIND OFF "Enable running of unit tests through valgrind")

    set(VALGRIND_COMMON_ARGS
            "--leak-check=full;--suppressions=${CMAKE_CURRENT_SOURCE_DIR}/combined_valgrind.supp;--error-exitcode=1")
    mark_as_advanced(VALGRIND)
endif (VALGRIND)

if (ENABLE_VALGRIND)
    set(DART_TESTING_TIMEOUT 4500 CACHE STRING "Testing timeout" FORCE)
endif (ENABLE_VALGRIND)

enable_testing()

###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
    test_aabb_tree
    test_convex_polygon
    test_convex_polyhedron
    test_ellipsoid
    test_faceted_sphere
    test_moves
    test_name
    test_polyhedron
    test_simple_polygon
    test_sphere
    test_sphere_union
    test_spheropolygon
    test_spheropolyhedron
    test_sphinx
    )

foreach (CUR_TEST ${TEST_LIST})
    # Need to define NO_IMPORT_ARRAY in every file but hoomd_module.cc
    set_source_files_properties(${CUR_TEST}.cc PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

    # add and link the unit test executable
    if(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
    cuda_add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu OPTIONS
            ${CUDA_ADDITIONAL_OPTIONS})
    else(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
    add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cc)
    endif(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)

    add_dependencies(test_all ${CUR_TEST})

    target_link_libraries(${CUR_TEST} ${ADDITIONAL_LIBS} ${HOOMD_COMMON_LIBS})
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
    get_target_property(CUR_TEST_EXE ${CUR_TEST} LOCATION)

    if (ENABLE_MPI)
        add_test(${CUR_TEST} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} ${CUR_TEST_EXE})
    else()
        add_test(${CUR_TEST} ${CUR_TEST_EXE})
    endif()
endforeach(CUR_TEST)
