CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

project (HOOMD LANGUAGES C CXX)

# bring in custom modules
add_subdirectory (CMake)

################################
## Version information
set(HOOMD_VERSION_RAW "2.6.0")
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
## Build options
option(SINGLE_PRECISION "Use single precision math" OFF)

option(ENABLE_HPMC_MIXED_PRECISION "Enable mixed precision computations in HPMC" ON)
if (ENABLE_HPMC_MIXED_PRECISION)
    add_definitions(-DENABLE_HPMC_MIXED_PRECISION)
endif()

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
option(BUILD_DEPRECATED "Build the deprecated package" on)
option(BUILD_METAL "Build the metal package" on)
option(BUILD_DEM "Build the dem package" on)
option(BUILD_CGCMM "Build the cgcmm package" on)
option(BUILD_MPCD "Build the mpcd package" on)
option(BUILD_JIT "Build the jit package" off)

set(HOOMD_COMMON_LIBS "")

#################################
## setup python library and executable
# Set default CFlags
include (HOOMDCFlagsSetup)
# setup MPI support
include (HOOMDMPISetup)
# find the python libraries to link to
include(HOOMDPythonSetup)
# Find CUDA and set it up
include (HOOMDCUDASetup)
# Find TBB and set it up
include (TBBSetup)

#################################
## Setup include directories and file lists for sub directories
include_directories(${HOOMD_SOURCE_DIR}
                    ${HOOMD_SOURCE_DIR}/hoomd/extern/cereal/include
                    ${CUDA_INCLUDE}
                    ${HOOMD_BINARY_DIR}/hoomd/include)

#######################################
# Global preprocessor flags
add_definitions(-D_REENTRANT)
add_definitions(-DEIGEN_MPL2_ONLY)

if (SINGLE_PRECISION)
    add_definitions (-DSINGLE_PRECISION)
# disable for now, only support single precision FFT
#else(SINGLE_PRECISION)
#   add_definitions (-Dkiss_fft_scalar=double)
endif(SINGLE_PRECISION)

if (APPLE)
    list(APPEND HOOMD_COMMON_LIBS "-undefined dynamic_lookup")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/libgetar)

#########################################
# Check for submodules
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/cereal/include/cereal/cereal.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/cub/cub/cub.cuh OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/nano-signal-slot/nano_signal_slot.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/pybind/include/pybind11/pybind11.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/upp11/upp11.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/quickhull/ConvexHull.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/random123/include/Random123/philox.h
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

################################
# install cmake scripts into hoomd/CMake

set(cmake_files CMake/hoomd/FindTBB.cmake
                CMake/hoomd/HOOMDCFlagsSetup.cmake
                CMake/hoomd/HOOMDCUDASetup.cmake
                CMake/hoomd/TBBSetup.cmake
                CMake/hoomd/test.cc
                CMake/hoomd/HOOMDMPISetup.cmake
                CMake/hoomd/HOOMDPythonSetup.cmake
                CMake/git/FindGit.cmake
                CMake/git/GetGitRevisionDescription.cmake
                CMake/libgetar/Findlibgetar.cmake
   )

# install headers in installation target
install(FILES ${cmake_files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

#######################
## Get the compile date
exec_program("date +%x" OUTPUT_VARIABLE COMPILE_DATE)

configure_file(hoomd_cache.cmake.in
    ${HOOMD_BINARY_DIR}/hoomd/CMake/hoomd/hoomd_cache.cmake)

install(FILES ${HOOMD_BINARY_DIR}/hoomd/CMake/hoomd/hoomd_cache.cmake
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/CMake/hoomd)

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
# this option turns off CPU only tests in builds with ENABLE_CUDA=ON
option(TEST_CPU_IN_GPU_BUILDS "Test CPU code path in GPU enabled builds" on)
mark_as_advanced(TEST_CPU_IN_GPU_BUILDS)
if (NOT TEST_CPU_IN_GPU_BUILDS AND ENABLE_CUDA)
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
