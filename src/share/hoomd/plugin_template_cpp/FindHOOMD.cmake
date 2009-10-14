# CMake script for finding HOOMD and setting up all needed compile options to create and link a plugin library
#
# Variables taken as input to this module:
# HOOMD_ROOT :          location to look for HOOMD, if it is not in the system path
#
# Variables defined by this module:
# FOUND_HOOMD :         set to true if HOOMD is found
# HOOMD_LIBRARIES :     a list of all libraries needed to link to to access hoomd
# HOOMD_INCLUDE_DIR :   a list of all include directories that need to be set to include HOOMD

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

# search for the hoomd include directory
find_path(HOOMD_INCLUDE_DIR
          NAMES hoomd_config.h
          HINTS ${HOOMD_ROOT}/include
          PATH_SUFFIXES hoomd
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
file(STRINGS ${HOOMD_INCLUDE_DIR}/hoomd_config.h _hoomd_config_h_lines)
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

endif (HOOMD_FOUND)