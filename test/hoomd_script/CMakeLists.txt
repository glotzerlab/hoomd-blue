# $Id$
# $ $URL$
# Maintainer: joaander

# setup hoomd runner environment
if (ENABLE_STATIC)
    get_target_property(HOOMD_STATIC_EXE hoomd LOCATION)
    set(HOOMD_RUN_EXE ${HOOMD_STATIC_EXE})
else (ENABLE_STATIC)
    get_target_property(LIBHOOMD_LOCATION libhoomd LOCATION)
    get_filename_component(LIBHOOMD_DIR ${LIBHOOMD_LOCATION} PATH)
    
    set(HOOMD_RUN_EXE ${HOOMD_BINARY_DIR}/python-runner/hoomd_run)
endif (ENABLE_STATIC)

if (CMAKE_MINOR_VERSION GREATER 7 AND ENABLE_STATIC)

add_test(NAME hoomd_script_tests_cpu COMMAND hoomd ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py "--mode=cpu")
if (ENABLE_CUDA)
add_test(NAME hoomd_script_tests_gpu COMMAND hoomd ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py "--mode=gpu")
endif (ENABLE_CUDA)

else (CMAKE_MINOR_VERSION GREATER 7 AND ENABLE_STATIC)

add_test(hoomd_script_tests_cpu ${HOOMD_RUN_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py "--mode=cpu")
if (ENABLE_CUDA)
add_test(hoomd_script_tests_gpu ${HOOMD_RUN_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py "--mode=gpu")
endif (ENABLE_CUDA)

endif (CMAKE_MINOR_VERSION GREATER 7 AND ENABLE_STATIC)
