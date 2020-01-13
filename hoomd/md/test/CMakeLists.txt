# Maintainer: joaander

###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
    test_berendsen_integrator
    test_bondtable_bond_force
    test_constraint_sphere
    test_dipole_force
    test_enforce2d_updater
    test_external_periodic
    test_fenebond_force
    test_fire_energy_minimizer
    test_force_shifted_lj
    test_gaussian_force
    test_gayberne_force
    test_cosinesq_angle_force
    test_harmonic_angle_force
    test_harmonic_bond_force
    test_harmonic_dihedral_force
    test_harmonic_improper_force
    test_lj_force
    test_mie_force
    test_morse_force
    test_MolecularForceCompute
    test_neighborlist
    test_opls_dihedral_force
    test_pppm_force
    test_slj_force
    test_table_angle_force
    test_table_dihedral_force
    test_table_potential
    test_temp_rescale_updater
    test_walldata
    test_yukawa_force
    test_zero_momentum_updater
    )

if (NOT HOOMD_SKIP_LONG_TESTS)
    # put the longest tests last
set(TEST_LIST ${TEST_LIST}
    test_dpd_integrator
    test_npt_mtk_integrator
    test_nvt_mtk_integrator
    test_nve_integrator)
endif()

if(ENABLE_MPI)
    MACRO(ADD_TO_MPI_TESTS _KEY _VALUE)
    SET("NProc_${_KEY}" "${_VALUE}")
    SET(MPI_TEST_LIST ${MPI_TEST_LIST} ${_KEY})
    ENDMACRO(ADD_TO_MPI_TESTS)

    # define every test together with the number of processors

    ADD_TO_MPI_TESTS(test_communication 8)
    ADD_TO_MPI_TESTS(test_communicator_grid 8)
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

    target_link_libraries(${CUR_TEST} _md ${HOOMD_LIBRARIES} ${PYTHON_LIBRARIES})
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
