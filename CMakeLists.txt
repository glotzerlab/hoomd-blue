CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)

if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0026 OLD)
    cmake_policy(SET CMP0042 OLD)
    set(CMAKE_MACOSX_RPATH ON)
endif(COMMAND cmake_policy)
endif()

project (HOOMD)

# bring in custom modules
add_subdirectory (CMake)
# setup version tagging info
include (CMake_version_options.cmake)

message(STATUS "Configuring HOOMD ${HOOMD_VERSION_LONG}")

# Handle user build options
include (CMake_build_options.cmake)

# setup the install directories
include (CMake_install_options.cmake)

#################################
## setup python library and executable
# setup MPI support
include (HOOMDMPISetup)
# find the python libraries to link to
include(HOOMDPythonSetup)
# Find CUDA and set it up
include (HOOMDCUDASetup)
# Set default CFlags
include (HOOMDCFlagsSetup)
# include some os specific options
include (HOOMDOSSpecificSetup)
# setup common libraries used by all targets in this project
include (HOOMDCommonLibsSetup)
# setup numpy
include (HOOMDNumpySetup)
# macros
include (HOOMDMacros)
# Configure some source files, include directories, and create variables listing all source files
include (CMake_src_setup.cmake)
# preprocessor flags
include (CMake_preprocessor_flags.cmake)

set (UPDATE_SUBMODULES 1 CACHE BOOL "Update git submodules for dependency projects")

if (UPDATE_SUBMODULES)
  message(STATUS "Updating git submodules")
  execute_process(COMMAND git submodule update --init
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else (UPDATE_SUBMODULES)
  message(STATUS "Not updating git submodules")
endif (UPDATE_SUBMODULES)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/libgetar)

find_package(libgetar)

if (libgetar_DIR)
  set(LIBGETAR_SHARED TRUE)
  add_subdirectory(${libgetar_DIR} EXCLUDE_FROM_ALL)
else (libgetar_DIR)
  message(FATAL_ERROR "Libgetar was not found in hoomd/extern/libgetar. Please "
                      "pull the libgetar source, i.e. via `git submodule update`.")
endif (libgetar_DIR)

################################
# set up unit tests
enable_testing()
option(BUILD_TESTING "Build unit tests" ON)
if (BUILD_TESTING)
     # add test_all to the ALL target
     add_custom_target(test_all ALL)
endif (BUILD_TESTING)

################################
## Process subdirectories
add_subdirectory (hoomd)

###############################
## include documentation directories
if (ENABLE_DOXYGEN)
    add_subdirectory (doc)
endif (ENABLE_DOXYGEN)
