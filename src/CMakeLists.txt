## $Id$
## $URL$
# Maintainer: joaander

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


project (HOOMD)
# Specify cmake modules path to use our custom modules
set(CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/python"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/cuda"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/hoomd"
    "${CMAKE_MODULE_PATH}")

# setup the install directories and setup HOOMD_VERSION
include (install_options.cmake)
message(STATUS "Configuring HOOMD ${HOOMD_VERSION_LONG}")

# Handle user build options
include (build_options.cmake)
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
# Configure some source files, include directories, and create variables listing all source files
include (src_setup.cmake)

# setup the library compile mode
if(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS OFF)
else(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS ON)
endif(ENABLE_STATIC)

################################
## Process subdirectories
add_subdirectory (lib)
add_subdirectory (microbenchmarks EXCLUDE_FROM_ALL)
add_subdirectory (share)
add_subdirectory (scripts)
add_subdirectory (python)
add_subdirectory (pluginapi)

################################
# set up unit tests
include(CTest)
mark_as_advanced(DART_TESTING_TIMEOUT)
if (BUILD_TESTING)
    add_subdirectory (unit_tests)
endif (BUILD_TESTING)

###############################
## include documentation directories
if (ENABLE_DOXYGEN)
    add_subdirectory (doc)
    add_subdirectory (user_doc)
endif (ENABLE_DOXYGEN)

add_subdirectory (installer)
