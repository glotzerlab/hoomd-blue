# CMake script for finding HOOMD and setting up all needed compile options to create and link a plugin library
#
# Variables taken as input to this module:
# HOOMD_ROOT :          location to look for HOOMD, if it is not in the python path
#
# Variables defined by this module:
# FOUND_HOOMD :         set to true if HOOMD is found
# HOOMD_LIBRARIES :     a list of all libraries needed to link to to access hoomd (uncached)
# HOOMD_INCLUDE_DIR :   a list of all include directories that need to be set to include HOOMD
# HOOMD_LIB :           a cached var locating the hoomd library to link to
#
# various ENABLE_ flags translated from hoomd_config.h so this plugin build can match the ABI of the installed hoomd
#
# as a convenience (for the intended purpose of this find script), all include directories and definitions needed
# to compile with all the various libs (boost, python, winsoc, etc...) are set within this script

set(HOOMD_ROOT "" CACHE FILEPATH "Directory containing a hoomd installation (i.e. _hoomd.so)")

# Let HOOMD_ROOT take precedence, but if unset, try letting Python find a hoomd package in its default paths.
if(HOOMD_ROOT)
  set(hoomd_installation_guess ${HOOMD_ROOT})
else(HOOMD_ROOT)
  find_package(PythonInterp)

  set(find_hoomd_script "
from __future__ import print_function;
import sys, os; sys.stdout = open(os.devnull, 'w')
import hoomd
print(os.path.dirname(hoomd.__file__), file=sys.stderr, end='')")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_hoomd_script}"
                  ERROR_VARIABLE hoomd_installation_guess)
  message(STATUS "Python output: " ${hoomd_installation_guess})
endif(HOOMD_ROOT)

message(STATUS "Looking for a HOOMD installation at " ${hoomd_installation_guess})
find_path(FOUND_HOOMD_ROOT
        NAMES _hoomd.so __init__.py
        HINTS ${hoomd_installation_guess}
        )

if(FOUND_HOOMD_ROOT)
  set(HOOMD_ROOT ${FOUND_HOOMD_ROOT} CACHE FILEPATH "Directory containing a hoomd installation (i.e. _hoomd.so)" FORCE)
  message(STATUS "Found hoomd installation at " ${HOOMD_ROOT})
else(FOUND_HOOMD_ROOT)
  message(FATAL_ERROR "Could not find hoomd installation, either set HOOMD_ROOT or set PYTHON_EXECUTABLE to a python which can find hoomd")
endif(FOUND_HOOMD_ROOT)

# search for the hoomd include directory
find_path(HOOMD_INCLUDE_DIR
          NAMES HOOMDVersion.h
          HINTS ${HOOMD_ROOT}/include
          )

if (HOOMD_INCLUDE_DIR)
    message(STATUS "Found HOOMD include directory: ${HOOMD_INCLUDE_DIR}")
    mark_as_advanced(HOOMD_INCLUDE_DIR)
endif (HOOMD_INCLUDE_DIR)

set(HOOMD_FOUND FALSE)
if (HOOMD_INCLUDE_DIR AND HOOMD_ROOT)
    set(HOOMD_FOUND TRUE)
    mark_as_advanced(HOOMD_ROOT)
endif (HOOMD_INCLUDE_DIR AND HOOMD_ROOT)

if (NOT HOOMD_FOUND)
    message(SEND_ERROR "HOOMD Not found. Please specify the location of your hoomd installation in HOOMD_ROOT")
endif (NOT HOOMD_FOUND)

#############################################################
## Now that we've found hoomd, lets do some setup
if (HOOMD_FOUND)

include_directories(${HOOMD_INCLUDE_DIR})

# run all of HOOMD's generic lib setup scripts
set(CMAKE_MODULE_PATH ${HOOMD_ROOT}
                      ${HOOMD_ROOT}/CMake/hoomd
                      ${HOOMD_ROOT}/CMake/thrust
                      ${CMAKE_MODULE_PATH}
                      )

# grab previously-set hoomd configuration
include (hoomd_cache)

# Handle user build options
include (CMake_build_options)
include (CMake_preprocessor_flags)
# setup the install directories
include (CMake_install_options)

# Find the python executable and libraries
include (HOOMDPythonSetup)
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

set(HOOMD_LIB ${HOOMD_ROOT}/_hoomd${PYTHON_MODULE_EXTENSION})
set(HOOMD_MD_LIB ${HOOMD_ROOT}/md/_md${PYTHON_MODULE_EXTENSION})
set(HOOMD_DEM_LIB ${HOOMD_ROOT}/dem/_dem${PYTHON_MODULE_EXTENSION})
set(HOOMD_HPMC_LIB ${HOOMD_ROOT}/hpmc/_hpmc${PYTHON_MODULE_EXTENSION})
set(HOOMD_CGCMM_LIB ${HOOMD_ROOT}/cgcmm/_cgcmm${PYTHON_MODULE_EXTENSION})
set(HOOMD_METAL_LIB ${HOOMD_ROOT}/metal/_metal${PYTHON_MODULE_EXTENSION})
set(HOOMD_DEPRECATED_LIB ${HOOMD_ROOT}/deprecated/_deprecated${PYTHON_MODULE_EXTENSION})

set(HOOMD_LIBRARIES ${HOOMD_LIB} ${HOOMD_COMMON_LIBS})
set(HOOMD_LIBRARIES ${HOOMD_LIB} ${HOOMD_COMMON_LIBS})

endif (HOOMD_FOUND)
