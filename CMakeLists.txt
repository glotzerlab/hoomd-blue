CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

project (HOOMD LANGUAGES C CXX)

# bring in custom modules
add_subdirectory (CMake)
# setup version tagging info
include (CMake_version_options.cmake)

message(STATUS "Configuring HOOMD ${HOOMD_VERSION_LONG}")

# Handle user build options
include (CMake_build_options.cmake)

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
# setup common libraries used by all targets in this project
include (HOOMDCommonLibsSetup)
# macros
include (HOOMDMacros)
# Configure some source files, include directories, and create variables listing all source files
include (CMake_src_setup.cmake)
# preprocessor flags
include (CMake_preprocessor_flags.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/libgetar)

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
                CMake/hoomd/HOOMDCommonLibsSetup.cmake
                CMake/hoomd/HOOMDCUDASetup.cmake
                CMake/hoomd/test.cc
                CMake/hoomd/HOOMDMacros.cmake
                CMake/hoomd/HOOMDMPISetup.cmake
                CMake/hoomd/HOOMDPythonSetup.cmake
                CMake/git/FindGit.cmake
                CMake/git/GetGitRevisionDescription.cmake
                CMake/libgetar/Findlibgetar.cmake
                CMake_build_options.cmake
                CMake_preprocessor_flags.cmake
                CMake_src_setup.cmake
                CMake_version_options.cmake
   )

# copy cmake scripts we currently use into build directory
MACRO(copy_cmake file)
    add_custom_command (
        OUTPUT hoomd/${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/hoomd/${file}
        COMMENT    "Copy hoomd/${file}"
    )
ENDMACRO(copy_cmake)

# install headers in installation target
install(FILES ${cmake_files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

# copy headers into build directory
foreach(file ${cmake_files})
    copy_cmake(${file})
endforeach()

configure_file(hoomd_cache.cmake.in
    ${HOOMD_BINARY_DIR}/hoomd/CMake/hoomd/hoomd_cache.cmake)

install(FILES ${HOOMD_BINARY_DIR}/hoomd/CMake/hoomd/hoomd_cache.cmake
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/CMake/hoomd)

add_custom_target(copy_cmake ALL
    DEPENDS ${cmake_files} ${HOOMD_BINARY_DIR}/hoomd/CMake/hoomd/hoomd_cache.cmake)

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

################################
## Process subdirectories
add_subdirectory (hoomd)

###############################
## include documentation directories
if (ENABLE_DOXYGEN)
    add_subdirectory (doc)
endif (ENABLE_DOXYGEN)
