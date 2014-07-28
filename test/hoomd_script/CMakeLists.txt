# Maintainer: joaander

#############################
# macro for adding hoomd script tests
macro(add_hoomd_script_test test_py)
# name the test
get_filename_component(_test_name ${test_py} NAME_WE)

if (CMAKE_MINOR_VERSION GREATER 7)
get_target_property(HOOMD_EXE hoomd LOCATION)
# with CMake 2.8.x and newer, use the NAME COMMAND form of add test to run the built hoomd target
add_test(NAME script-${_test_name}-cpu COMMAND ${HOOMD_EXE} ${test_py} "--mode=cpu" "--gpu_error_checking")
if (ENABLE_CUDA)
add_test(NAME script-${_test_name}-gpu COMMAND ${HOOMD_EXE} ${test_py} "--mode=gpu" "--gpu_error_checking")
endif (ENABLE_CUDA)

else (CMAKE_MINOR_VERSION GREATER 7)
# with CMake 2.6.x, or shared builds use the old form of add_test to add a specific hoomd exename
# setup hoomd runner environment
get_target_property(HOOMD_EXE hoomd LOCATION)

add_test(script-${_test_name}-cpu ${HOOMD_EXE} ${test_py} "--mode=cpu" "--gpu_error_checking")
if (ENABLE_CUDA)
add_test(script-${_test_name}-gpu ${HOOMD_EXE} ${test_py} "--mode=gpu" "--gpu_error_checking")
endif (ENABLE_CUDA)

endif (CMAKE_MINOR_VERSION GREATER 7)
endmacro(add_hoomd_script_test)
###############################

#############################
# macro for adding hoomd script tests (MPI version)
macro(add_hoomd_script_test_mpi test_py nproc)
# name the test
get_filename_component(_test_name ${test_py} NAME_WE)
get_target_property(HOOMD_EXE hoomd LOCATION)

add_test(NAME script-${_test_name}-mpi-cpu
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
         ${MPIEXEC_POSTFLAGS} ${HOOMD_EXE} ${test_py} "--mode=cpu" "--gpu_error_checking")
if (ENABLE_CUDA)
add_test(NAME script-${_test_name}-mpi-gpu
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
         ${MPIEXEC_POSTFLAGS} ${HOOMD_EXE} ${test_py} "--mode=gpu" "--gpu_error_checking")
endif (ENABLE_CUDA)
endmacro(add_hoomd_script_test_mpi)
###############################


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configure.ocelot ${CMAKE_CURRENT_BINARY_DIR}/configure.ocelot COPYONLY)

# loop through all test_*.py files
file(GLOB _hoomd_script_tests ${CMAKE_CURRENT_SOURCE_DIR}/test_*.py)

foreach(test ${_hoomd_script_tests})
add_hoomd_script_test(${test})
endforeach(test)

# exclude some tests from MPI
SET(EXCLUDE_FROM_MPI
    test_charge_pppm
    test_constraint_sphere
    test_dump_mol2
    test_dump_pdb
    test_integrate_bdnvt_rigid
    test_integrate_nvt_rigid
    test_integrate_nve_rigid
    test_pair_cgcmm
    test_update_rescale_temp
    test_wall_lj
    )

if (ENABLE_MPI)
    foreach(test ${_hoomd_script_tests})
        GET_FILENAME_COMPONENT(test_name ${test} NAME_WE)
        if(NOT "${EXCLUDE_FROM_MPI}" MATCHES ${test_name})
            # execute on two processors
            add_hoomd_script_test_mpi(${test} 2)
        endif()
    endforeach(test)
endif(ENABLE_MPI)
