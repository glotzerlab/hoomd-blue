# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

## $Id$
## $URL$

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


project (HOOMD)
# Specify cmake modules path to use our custom modules
set(CMAKE_MODULE_PATH 
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake/boost"
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake/cuda"
	"${CMAKE_MODULE_PATH}")
	
# setup flags to specify installation directories for files, these differ in
# linux and windows
if (WIN32)
	# The "." needs to be there to install to the root directory of
	# the specified install path, "" doesn't work
	set(DATA_INSTALL_DIR ".")
	set(LIB_INSTALL_DIR "bin")
	set(BIN_INSTALL_DIR "bin")
else (WIN32)
	set(DATA_INSTALL_DIR "share/hoomd")
	set(LIB_INSTALL_DIR "lib/hoomd")
	set(BIN_INSTALL_DIR "bin")
endif (WIN32)
 

################################
## Version information
set(HOOMD_VERSION_MAJOR "0")
set(HOOMD_VERSION_MINOR "8")
set(HOOMD_VERSION_PATCH "0")
set(HOOMD_VERSION "${HOOMD_VERSION_MAJOR}.${HOOMD_VERSION_MINOR}.${HOOMD_VERSION_PATCH}")
set(HOOMD_SUBVERSION_BUILD "1")

# users may not have subversion installed, search for it and set an unknown version if that is
# the case
find_program(SVNVERSION_EXE "svnversion" DOC "svnversion executable")
mark_as_advanced(SVNVERSION_EXE)
if (SVNVERSION_EXE)
	exec_program(${SVNVERSION_EXE} ${HOOMD_SOURCE_DIR} ARGS ${HOOMD_SOURCE_DIR} OUTPUT_VARIABLE SVNVERSION)
else (SVNVERSION_EXE)
	set(SVNVERSION "NA")
endif (SVNVERSION_EXE)

if(HOOMD_SUBVERSION_BUILD)
	string(REGEX REPLACE ":" "_" SVNVERSION_UNDERSCORE ${SVNVERSION})
	set(CPACK_PACKAGE_VERSION "${HOOMD_VERSION}_beta${SVNVERSION_UNDERSCORE}")
endif(HOOMD_SUBVERSION_BUILD)


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

#################################
## Name the output targets

# name of the python module and hoomd executable targets
set(HOOMD_PYTHON_MODULE hoomd_python_module)
set(HOOMD_EXE hoomd_exe)

#################################
## setup python library and executable
# find the python interpreter, first
find_package(PythonInterp REQUIRED)
# find the python libraries to link to
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
add_subdirectory (python)

################################
# set up unit tests
enable_testing()
add_subdirectory (unit_tests)

################################
## Process subdirectories
add_subdirectory (lib)
add_subdirectory (benchmarks)
add_subdirectory (share)
add_subdirectory (scripts)

###############################
## allow for optional inclusion of the doxygen build
OPTION(USE_DOXYGEN "Enable generation of source code documentation with doxygen (if installed)" OFF)
if(USE_DOXYGEN)
	add_subdirectory (doc)
	add_subdirectory (user_doc)
endif(USE_DOXYGEN)

add_subdirectory (installer)

###############################
# allow library linking for MacOS10.5
if (APPLE)
add_definitions(-mmacosx-version-min=10.4)
endif (APPLE)
