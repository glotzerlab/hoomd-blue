CMAKE_MINIMUM_REQUIRED(VERSION 3.15...3.27 FATAL_ERROR)
# >= 3.8 is required for CUDA language support
# >= 3.9 is required for MPI::MPI_CXX target

# Search for Python and other libraries in unix-like locations first and frameworks last.
# This allows FindPython to find virtual environment Pythons before a homebrew or system Pythons.
set(CMAKE_FIND_FRAMEWORK LAST)

project (HOOMD LANGUAGES C CXX)

# enlble compile_commands.json output
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

# bring in custom modules
add_subdirectory (CMake)

################################
## Version information
set(HOOMD_VERSION_RAW "4.8.2")
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

# enable c++17
# Since we support the last two major releases of the CUDA toolkit, we cannot
# use C++17 as the CUDA standard until CUDA 12.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_HIP_STANDARD 14)

# Enable compiler warnings on gcc and clang (common compilers used by developers)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result")
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
set(HOOMD_SHORTREAL_SIZE "32" CACHE STRING "Size of the ShortReal type in bits.")
SET_PROPERTY(CACHE HOOMD_SHORTREAL_SIZE PROPERTY STRINGS "32" "64")
set(HOOMD_LONGREAL_SIZE "64" CACHE STRING "Size of the LongReal type in bits.")
SET_PROPERTY(CACHE HOOMD_LONGREAL_SIZE PROPERTY STRINGS "32" "64")
OPTION(ENABLE_GPU "True if we are compiling for a GPU target" FALSE)
SET(ENABLE_HIP ${ENABLE_GPU})
set(HOOMD_GPU_PLATFORM "CUDA" CACHE STRING "Choose the GPU backend: HIP or CUDA.")
option(ENABLE_LLVM "Link to the LLVM library for run time code generation" off)

# Components
option(BUILD_MD "Build the md package" on)
if (HOOMD_LONGREAL_SIZE STREQUAL "64")
option(BUILD_HPMC "Build the hpmc package" on)
else ()
option(BUILD_HPMC "Build the hpmc package" off)
endif()
option(BUILD_METAL "Build the metal package" on)

if (ENABLE_GPU AND HOOMD_GPU_PLATFORM STREQUAL "HIP")
message("Defaulting BUILD_MPCD=off due to HIP GPU platform.")
option(BUILD_MPCD "Build the mpcd package" off)
else()
option(BUILD_MPCD "Build the mpcd package" ${BUILD_MD})
endif()

# Optionally use TBB for threading
option(ENABLE_TBB "Enable support for Threading Building Blocks (TBB)" off)

# Add list of plugins
set(PLUGINS "example_plugins/pair_plugin;example_plugins/updater_plugin;example_plugins/shape_plugin" CACHE STRING "List of plugin directories.")

# this needs to go before CUDA setup
include (HOOMDHIPSetup)

# Find CUDA and set it up
include (HOOMDCUDASetup)

# setup MPI support
include (HOOMDMPISetup)
# find the python libraries to link to
include(HOOMDPythonSetup)

include (hoomd-macros)

find_package(Eigen3 3.2 CONFIG REQUIRED)
if (Eigen3_FOUND)
    find_package_message(EIGEN3 "Found eigen: ${Eigen3_DIR} ${EIGEN3_INCLUDE_DIR} (version ${Eigen3_VERSION})" "[${Eigen3_DIR}][${EIGEN3_INCLUDE_DIR}]")
endif()

#########################################
# Check for submodules
if (
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/nano-signal-slot/nano_signal_slot.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/upp11/upp11.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/HIP/include/hip/hip_runtime.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/quickhull/ConvexHull.hpp OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/random123/include/Random123/philox.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/neighbor/include/neighbor/neighbor.h OR
    NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/hipper/include/hipper/hipper_runtime.h
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

#######################
## Get the compile date
execute_process(COMMAND "date" "+%Y-%m-%d" OUTPUT_VARIABLE COMPILE_DATE OUTPUT_STRIP_TRAILING_WHITESPACE)

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
              CMake/hoomd/HOOMDHIPSetup.cmake
              CMake/hoomd/hoomd-macros.cmake
              ${HOOMD_BINARY_DIR}/hoomd-config.cmake
        DESTINATION ${CONFIG_INSTALL_DIR})

# enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT WIN32)
  file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC
  )
endif()
