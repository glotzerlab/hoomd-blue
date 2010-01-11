# $Id$
# $ $URL$
# Maintainer: joaander

###############################
# UNIX like OSes need a few settings for the unit test library
if (NOT WIN32)
    # first, they need the boost unit test framework added as a library
    set(ADDITIONAL_LIBS libhoomd
            ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
            )

    # then we need to specify BOOST_TEST_DYN_LINK if the unit test framework library is a shared lib
    if("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.dylib$")
    add_definitions(-DBOOST_TEST_DYN_LINK)
    endif("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.dylib$")
    
    if("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")
    add_definitions(-DBOOST_TEST_DYN_LINK)
    endif("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")
endif (NOT WIN32)

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
    test_bd_rigid_updater
    test_index1d
    test_molfile
    test_gpu_array
    test_pdata
    test_particle_group
    test_utils
    test_harmonic_bond_force
    test_harmonic_angle_force
    test_harmonic_dihedral_force
    test_harmonic_improper_force
    test_cgcmm_angle_force
    test_fenebond_force
    test_lj_wall_force
    test_neighborlist
    test_lj_force
    test_table_potential
    test_slj_force
    test_gaussian_force
    test_yukawa_force
    test_cgcmm_force
    test_morse_force
    test_nve_integrator
    test_nvt_integrator
    test_zero_momentum_updater
    test_temp_rescale_updater
    test_hoomd_xml
    test_system
    test_binary_reader_writer
    test_enforce2d_updater
    )

if(ENABLE_CUDA)
    set(TEST_LIST ${TEST_LIST}
    test_gpu_worker)
endif(ENABLE_CUDA)

    # put the longest tests last
    set(TEST_LIST ${TEST_LIST}
    test_bdnvt_integrator
    test_npt_integrator)

foreach (CUR_TEST ${TEST_LIST})
    # add and link the unit test executable
    if(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
    cuda_add_executable(${CUR_TEST} ${CUR_TEST}.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu OPTIONS
            ${CUDA_ADDITIONAL_OPTIONS})
    else(ENABLEC_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
    add_executable(${CUR_TEST} ${CUR_TEST}.cc)
    endif(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)

    add_dependencies(test_all ${CUR_TEST})
    
    target_link_libraries(${CUR_TEST} libhoomd ${ADDITIONAL_LIBS} ${HOOMD_COMMON_LIBS})
    fix_cudart_rpath(${CUR_TEST})

    # add it to the unit test list
    get_target_property(CUR_TEST_EXE ${CUR_TEST} LOCATION)
    
    # unfortunately, this needs to be handled differently in visual studio
    if (MSVC)
        add_test(${CUR_TEST} ${EXECUTABLE_OUTPUT_PATH}/${CUR_TEST})
    else (MSVC)
    # AND it needs to be handled differently for XCODE
    if (CMAKE_GENERATOR MATCHES "Xcode")
        add_test(${CUR_TEST} ${CUR_TEST})
    else (CMAKE_GENERATOR MATCHES "Xcode")
        add_test(${CUR_TEST} ${CUR_TEST_EXE})
        # add a valgrind executed test if requested
        if (ENABLE_VALGRIND)
            add_test(valgrind_${CUR_TEST} ${VALGRIND} ${VALGRIND_COMMON_ARGS} ${CUR_TEST_EXE})
        endif (ENABLE_VALGRIND)
    endif (CMAKE_GENERATOR MATCHES "Xcode")
    endif (MSVC)
endforeach (CUR_TEST)
