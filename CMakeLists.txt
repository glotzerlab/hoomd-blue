CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)
# >= 3.8 is required for CUDA language support
# >= 3.9 is required for MPI::MPI_CXX target

project (HOOMD LANGUAGES C CXX)

# enlble compile_commands.json output
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

# bring in custom modules
add_subdirectory (CMake)

################################
## Version information
set(HOOMD_VERSION_RAW "2.9.0")
string(REGEX MATCH "(.*)\\.(.*)\\.(.*)$" _hoomd_version_match ${HOOMD_VERSION_RAW})
set(HOOMD_VERSION_MAJOR ${CMAKE_MATCH_1})
set(HOOMD_VERSION_MINOR ${CMAKE_MATCH_2})
set(HOOMD_VERSION_PATCH ${CMAKE_MATCH_3})
set(HOOMD_VERSION "${HOOMD_VERSION_MAJOR}.${HOOMD_VERSION_MINOR}.${HOOMD_VERSION_PATCH}")

# users may not have git installed, or this may be a tarball build - set a dummy version if that is the case
include(GetGitRevisionDescription)
git_describe(HOOMD_GIT_VERSION)
if (HOOMD_GIT_VERSION)
    set(HOOMD_VERSION_LONG "${HOOMD_GIT_VERSION}")
else (HOOMD_GIT_VERSION)
    set(HOOMD_VERSION_LONG "${HOOMD_VERSION}")
endif (HOOMD_GIT_VERSION)

get_git_head_revision(GIT_REFSPEC GIT_SHA1)
if (GIT_REFSPEC)
    set(HOOMD_GIT_REFSPEC "${GIT_REFSPEC}")
else (GIT_REFSPEC)
    set(HOOMD_GIT_REFSPEC "${HOOMD_VERSION_RAW}")
endif (GIT_REFSPEC)

if (GIT_SHA1)
    set(HOOMD_GIT_SHA1 "${GIT_SHA1}")
else (GIT_SHA1)
    set(HOOMD_GIT_SHA1 "unknown")
endif (GIT_SHA1)

message(STATUS "Configuring HOOMD ${HOOMD_VERSION_LONG}")

#################################
## CFLAGS configuration

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# enable c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)

# Enable compiler warnings on gcc and clang (common compilers used by developers)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
endif()

# Enable color output from compiler
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
endif()

#################################
## Build options
option(SINGLE_PRECISION "Use single precision math" OFF)
OPTION(ENABLE_GPU "True if we are compiling for a GPU target" FALSE)
SET(ENABLE_HIP ${ENABLE_GPU})

option(ENABLE_HPMC_MIXED_PRECISION "Enable mixed precision computations in HPMC" ON)

# Optionally enable documentation build
OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" OFF)
if (ENABLE_DOXYGEN)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # get the doxygen version
        exec_program(${DOXYGEN_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS --version OUTPUT_VARIABLE DOXYGEN_VERSION)

        if (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
        else (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
            message(STATUS "Doxygen version less than 1.8.5, documentation may not build correctly")
        endif (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
    endif ()
endif ()

# Components
option(BUILD_MD "Build the md package" on)
if (NOT SINGLE_PRECISION)
option(BUILD_HPMC "Build the hpmc package" on)
else ()
option(BUILD_HPMC "Build the hpmc package" off)
endif()
option(BUILD_METAL "Build the metal package" on)
option(BUILD_DEM "Build the dem package" on)
option(BUILD_JIT "Build the jit package" off)

# Optionally use TBB for threading
option(ENABLE_TBB "Enable support for Threading Building Blocks (TBB)" off)

# this needs to go before CUDA setup
include (HOOMDHIPSetup)

# Find CUDA and set it up
include (HOOMDCUDASetup)
include (hoomd-macros)

# setup MPI support
include (HOOMDMPISetup)
# find the python libraries to link to
include(HOOMDPythonSetup)

if (NOT ENABLE_HIP OR HIP_PLATFORM STREQUAL "nvcc")
    option(BUILD_MPCD "Build the mpcd package" on)
else()
    option(BUILD_MPCD "Build the mpcd package" off)
endif()

find_package(Eigen3 3.2 CONFIG REQUIRED)
if (Eigen3_FOUND)
    find_package_message(EIGEN3 "Found eigen: ${Eigen3_DIR} ${EIGEN3_INCLUDE_DIR} (version ${Eigen3_VERSION})" "[${Eigen3_DIR}][${EIGEN3_INCLUDE_DIR}]")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/libgetar)

#########################################
# Check for submodules
if (
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/cub/cub/cub.cuh OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/nano-signal-slot/nano_signal_slot.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/upp11/upp11.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/HIP/include/hip/hip_runtime.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/quickhull/ConvexHull.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/random123/include/Random123/philox.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/neighbor/neighbor/LBVH.h
    )

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
        # appropriate message for a git repository
        message(FATAL_ERROR "Submodules not found. Execute `git submodule update --init`. "
                            "in the source directory to provide them, or clone with the --recursive option.")
    else()
        # appropriate message for a tarball checkout
        message(FATAL_ERROR "Submodules not found. This is not a git clone. You can either use git to clone hoomd "
                            "or you can manually download all the required submodules and extract them in the proper "
                            "location in `hoomd/extern`. See the file .gitmodules for a list of all submodules "
                            "and the hoomd git repository submodule references for which commits of these repositories "
                            "must be provided.")
    endif()
endif()

find_package(libgetar)

if (libgetar_DIR)
  set(LIBGETAR_SHARED TRUE)
  add_subdirectory(${libgetar_DIR} EXCLUDE_FROM_ALL)
else (libgetar_DIR)
  message(FATAL_ERROR "Libgetar was not found in hoomd/extern/libgetar. Please "
                      "pull the libgetar source, i.e. via `git submodule update`.")
endif (libgetar_DIR)

#######################
## Get the compile date
exec_program("date +%x" OUTPUT_VARIABLE COMPILE_DATE)

################################
# set up unit tests
enable_testing()
option(BUILD_TESTING "Build unit tests" ON)

################################
# set up long validation tests
option(BUILD_VALIDATION "Build validation tests" OFF)
if (BUILD_TESTING OR BUILD_VALIDATION)
     # add test_all to the ALL target
     add_custom_target(test_all ALL)
endif (BUILD_TESTING OR BUILD_VALIDATION)

# In jenkins tests on multiple build configurations, it is wasteful to run CPU tests on CPU and all GPU test paths
# this option turns off CPU only tests in builds with ENABLE_HIP=ON
option(TEST_CPU_IN_GPU_BUILDS "Test CPU code path in GPU enabled builds" on)
mark_as_advanced(TEST_CPU_IN_GPU_BUILDS)
if (NOT TEST_CPU_IN_GPU_BUILDS AND ENABLE_HIP)
    message(STATUS "Warning: Skipping CPU tests")
endif()

################################
## Process subdirectories
add_subdirectory (hoomd)

###############################
## include documentation directories
if (ENABLE_DOXYGEN)
    add_subdirectory (doc)
endif (ENABLE_DOXYGEN)

###############################
## install cmake config files

include(CMakePackageConfigHelpers)

# version information
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hoomd-config-version.cmake
  VERSION ${HOOMD_VERSION}
  COMPATIBILITY SameMajorVersion)

# exports
export(EXPORT HOOMDTargets
       NAMESPACE "HOOMD::"
       FILE "${CMAKE_CURRENT_BINARY_DIR}/hoomd-targets.cmake")

set(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/hoomd")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hoomd-config-version.cmake
        DESTINATION ${CONFIG_INSTALL_DIR})

install(EXPORT HOOMDTargets
  NAMESPACE "HOOMD::"
  FILE hoomd-targets.cmake
  DESTINATION ${CONFIG_INSTALL_DIR})

configure_package_config_file(hoomd-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hoomd-config.cmake
  INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
  PATH_VARS CMAKE_INSTALL_PREFIX)

install(FILES CMake/hoomd/FindTBB.cmake
              CMake/hoomd/FindCUDALibs.cmake
              CMake/HIP/FindHIP.cmake
              CMake/hoomd/HOOMDHIPSetup.cmake
              CMake/hoomd/hipcc.cmake
              CMake/hoomd/hoomd-macros.cmake
              ${HOOMD_BINARY_DIR}/hoomd-config.cmake
        DESTINATION ${CONFIG_INSTALL_DIR})

install(FILES CMake/HIP/FindHIP/run_hipcc.cmake
              CMake/HIP/FindHIP/run_make2cmake.cmake
        DESTINATION ${CONFIG_INSTALL_DIR}/FindHIP)
