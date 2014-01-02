# CMake script for finding HOOMD and setting up all needed compile options to create and link a plugin library
#
# Variables taken as input to this module:
# HOOMD_ROOT :          location to look for HOOMD, if it is not in the system path
#
# Variables defined by this module:
# FOUND_HOOMD :         set to true if HOOMD is found
# HOOMD_LIBRARIES :     a list of all libraries needed to link to to access hoomd (uncached)
# HOOMD_INCLUDE_DIR :   a list of all include directories that need to be set to include HOOMD
# HOOMD_BIN_DIR :       the directory containing the hoomd runner executable
# HOOMD_LIB :           a cached var locating the hoomd library to link to
#
# various ENABLE_ flags translated from hoomd_config.h so this plugin build can match the ABI of the installed hoomd
#
# as a convenience (for the intended purpose of this find script), all include directories and definitions needed
# to compile with all the various libs (boost, python, winsoc, etc...) are set within this script

# see if we can find the HOOMD bin/ directory first. This usually works well if "hoomd" is in the path
find_path(HOOMD_BIN_DIR
          NAMES hoomd
          )

set(_hoomd_root_guess "HOOMD_ROOT-NOTFOUND")
if (HOOMD_BIN_DIR)
    message(STATUS "Found HOOMD bin directory: ${HOOMD_BIN_DIR}")
    mark_as_advanced(HOOMD_BIN_DIR)
    # guess the root dir location from the bin
    string(REGEX REPLACE "[/\\\\]?bin*[/\\\\]?$" "" _hoomd_root_guess ${HOOMD_BIN_DIR})
endif (HOOMD_BIN_DIR)

# root directory where HOOMD was found
set(HOOMD_ROOT ${_hoomd_root_guess} CACHE PATH "Root directory where HOOMD is installed")

# try again to find HOOMD_BIN_DIR
if (NOT HOOMD_BIN_DIR)
    find_path(HOOMD_BIN_DIR
              NAMES hoomd
              HINTS ${HOOMD_ROOT}/bin
              )
endif (NOT HOOMD_BIN_DIR)

if (HOOMD_BIN_DIR)
    message(STATUS "Found HOOMD bin directory: ${HOOMD_BIN_DIR}")
endif (HOOMD_BIN_DIR)

# search for the hoomd include directory
find_path(HOOMD_INCLUDE_DIR
          NAMES hoomd/hoomd_config.h
          HINTS ${HOOMD_ROOT}/include
          )

if (HOOMD_INCLUDE_DIR)
    message(STATUS "Found HOOMD include directory: ${HOOMD_INCLUDE_DIR}")
    mark_as_advanced(HOOMD_INCLUDE_DIR)
endif (HOOMD_INCLUDE_DIR)

# find the hoomd library
# we need to add a blank prefix to find the python module library
set(_old_prefixes ${CMAKE_FIND_LIBRARY_PREFIXES})
set(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
find_library(HOOMD_LIB
             NAMES hoomd
             HINTS ${HOOMD_ROOT}/lib/hoomd/python-module ${HOOMD_ROOT}/lib
                   ${HOOMD_ROOT}/lib64/hoomd/python-module ${HOOMD_ROOT}/lib64
                   ${HOOMD_ROOT}/lib64/python/site-packages
                   ${HOOMD_ROOT}/lib64/python2.4/site-packages
                   ${HOOMD_ROOT}/lib64/python2.5/site-packages
                   ${HOOMD_ROOT}/lib64/python2.6/site-packages
                   ${HOOMD_ROOT}/lib64/python2.7/site-packages
                   ${HOOMD_ROOT}/lib/python/site-packages
                   ${HOOMD_ROOT}/lib/python2.4/site-packages
                   ${HOOMD_ROOT}/lib/python2.5/site-packages
                   ${HOOMD_ROOT}/lib/python2.6/site-packages
                   ${HOOMD_ROOT}/lib/python2.7/site-packages
             )
set(CMAKE_FIND_LIBRARY_PREFIXES ${_old_prefixes})

if (HOOMD_LIB)
    message(STATUS "Found HOOMD library: ${HOOMD_LIB}")
    mark_as_advanced(HOOMD_LIB)
endif (HOOMD_LIB)

set(HOOMD_FOUND FALSE)
if (HOOMD_INCLUDE_DIR AND HOOMD_BIN_DIR AND HOOMD_ROOT AND HOOMD_LIB)
    set(HOOMD_FOUND TRUE)
    mark_as_advanced(HOOMD_ROOT)
endif (HOOMD_INCLUDE_DIR AND HOOMD_BIN_DIR AND HOOMD_ROOT AND HOOMD_LIB)

if (NOT HOOMD_FOUND)
    message(SEND_ERROR "HOOMD Not found. Please specify the location of your hoomd installation in HOOMD_ROOT")
endif (NOT HOOMD_FOUND)

#############################################################
## Now that we've found hoomd, lets do some setup
if (HOOMD_FOUND)

# read in hoomd_config.h
file(STRINGS ${HOOMD_INCLUDE_DIR}/hoomd/hoomd_config.h _hoomd_config_h_lines)
# go through all the lines in the file
foreach(_line ${_hoomd_config_h_lines})
    # if this line is #define VARIABLE
    if (${_line} MATCHES "^#define .*$")
        string(REGEX REPLACE "#define (.*)$" "\\1" _var ${_line})
        # and if it is not HOOMD_CONFIG_H
        if (NOT ${_var} MATCHES "HOOMD_CONFIG_H")
            message(STATUS "found define: ${_var}")
            # translate it to a CMake cache variable
            set(${_var} ON CACHE BOOL "Imported setting from hoomd_config.h, it matches the setting used to build HOOMD" FORCE)
        endif (NOT ${_var} MATCHES "HOOMD_CONFIG_H")
    endif (${_line} MATCHES "^#define .*$")

    # if this line is an #undef VARIABLE
    if (${_line} MATCHES "#undef .*")
        string(REGEX REPLACE "/. #undef (.*) ./" "\\1" _var ${_line})
        message(STATUS "found undef: ${_var}")
        # translate it to a CMake cache variable
        set(${_var} OFF CACHE BOOL "Imported setting from hoomd_config.h, it matches the setting used to build HOOMD" FORCE)
    endif (${_line} MATCHES "#undef .*")
endforeach()

include_directories(${HOOMD_INCLUDE_DIR})
include_directories(${HOOMD_INCLUDE_DIR}/hoomd)

# run all of HOOMD's generic lib setup scripts
set(CMAKE_MODULE_PATH ${HOOMD_ROOT}/share/hoomd/CMake/cuda
                      ${HOOMD_ROOT}/share/hoomd/CMake/hoomd
                      ${HOOMD_ROOT}/share/hoomd/CMake/python
                      ${CMAKE_MODULE_PATH}
                      )

# Find the python executable and libraries
include (HOOMDPythonSetup)
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
# setup macros
include (HOOMDMacros)
# setup MPI support
include (HOOMDMPISetup)

set(HOOMD_LIBRARIES ${HOOMD_LIB} ${HOOMD_COMMON_LIBS})

endif (HOOMD_FOUND)
