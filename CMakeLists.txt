# Maintainer: joaander

if (WIN32)
# require cmake 2.8 for add_test(NAME COMMAND) form on windows
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)
else (WIN32)
# require cmake 2.6.2 for findCUDA.cmake
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)
endif (WIN32)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project (HOOMD)

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR "The build directory must be outside of the source directory")
endif(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

# setup the install directories
include (CMake_install_options.cmake)
# bring in custom modules
add_subdirectory (CMake)
# setup version tagging info
include (CMake_version_options.cmake)

message(STATUS "Configuring HOOMD ${HOOMD_VERSION_LONG}")

# Handle user build options
include (CMake_build_options.cmake)
# Find the boost libraries and set them up
include (HOOMDBoostSetup)
# Find CUDA and set it up
include (HOOMDCUDASetup)
# Set default CFlags
include (HOOMDCFlagsSetup)
# include some os specific options
include (HOOMDOSSpecificSetup)
# setup common libraries used by all targets in this project
include (HOOMDCommonLibsSetup)
# setup MPI support
include (HOOMDMPISetup)
# macros
include (HOOMDMacros)
# Configure some source files, include directories, and create variables listing all source files
include (CMake_src_setup.cmake)
# preprocessor flags
include (CMake_preprocessor_flags.cmake)

# setup the library compile mode
if(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS OFF)
else(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS ON)
endif(ENABLE_STATIC)

################################
## Process subdirectories
add_subdirectory (libhoomd)
add_subdirectory (share)
add_subdirectory (python-module)
add_subdirectory (python-runner)

###############################
## include documentation directories
if (ENABLE_DOXYGEN)
    add_subdirectory (doc)
endif (ENABLE_DOXYGEN)

################################
# set up unit tests
include(CTest)
mark_as_advanced(DART_TESTING_TIMEOUT)
if (BUILD_TESTING)
    # add test_all to the ALL target if requested (mainly for automatic test
    # suites)
    if (ENABLE_TEST_ALL)
        set(_default_test_all ALL)
    endif (ENABLE_TEST_ALL)
    # setup convenience targets that prevent tests from always being built    
    add_custom_target(test_all ${_default_test_all})
    add_dependencies(test_all hoomd)
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
    add_dependencies(check test_all)
    add_subdirectory (test)
endif (BUILD_TESTING)

add_subdirectory (packaging)

