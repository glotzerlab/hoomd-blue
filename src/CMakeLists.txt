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
    "${CMAKE_MODULE_PATH}")
    

# setup the install directories and setup HOOMD_VERSION
include (CMakeInstSetup.txt)

message(STATUS "Configuring HOOMD ${HOOMD_VERSION}")

# Setup a number of misc options and libraries
include (CMakeMiscSetup.txt)
# Find the boost libraries and set them up
include (CMakeBoostSetup.txt)
# Find CUDA and set it up
include (CMakeCUDASetup.txt)
# Set default CFlags
include (CMakeCFlagsSetup.txt)
# Configure some source files, include directories, and create variables listing all source files
include (CMakeSRCSetup.txt)
# Configure the FFT libraries
include (CMakeFFTSetup.txt)

################################
## Define common libraries used by every target in HOOMD
if (WIN32)
    set(HOOMD_COMMON_LIBS hoomd_python_module ${FFTW_LIBRARY} ${PYTHON_LIBRARIES} ${WINSOCK_LIB})
else (WIN32)
    set(BOOST_LIBS ${Boost_THREAD_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_PYTHON_LIBRARY}
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            ${Boost_SIGNALS_LIBRARY}
            )
    if (Boost_SYSTEM_LIBRARY)
        set(BOOST_LIBS ${BOOST_LIBS} ${Boost_SYSTEM_LIBRARY})
    endif (Boost_SYSTEM_LIBRARY)

    ## An update to to CentOS5's python broke linking of the hoomd exe. According
    ## to an ancient post online, adding -lutil fixed this in python 2.2
    set(ADDITIONAL_LIBS "")
    if (UNIX AND NOT APPLE)
        find_library(UTIL_LIB util /usr/lib)
        find_library(DL_LIB dl /usr/lib)
        set(ADDITIONAL_LIBS ${UTIL_LIB} ${DL_LIB})
        if (DL_LIB AND UTIL_LIB)
        mark_as_advanced(UTIL_LIB DL_LIB)
        endif (DL_LIB AND UTIL_LIB)
    endif (UNIX AND NOT APPLE)

    set(HOOMD_COMMON_LIBS
            ${BOOST_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${FFTW_LIBRARY}
            ${PYTHON_LIBRARIES}
            ${ADDITIONAL_LIBS}
            )
endif (WIN32)

# setup the library compile mode
if(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS OFF)
    add_definitions(-DBOOST_PYTHON_STATIC_LIB)
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

################################
# set up unit tests
include(CTest)
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

###############################
# allow library linking for MacOS10.5
if (APPLE)
add_definitions(-mmacosx-version-min=10.4)
endif (APPLE)
