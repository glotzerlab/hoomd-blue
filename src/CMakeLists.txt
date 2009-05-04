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
# Maintainer: joaander

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
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
	set(HOOMD_LIB_MODE STATIC)
	add_definitions(-DBOOST_PYTHON_STATIC_LIB)
else(ENABLE_STATIC)
	set(HOOMD_LIB_MODE SHARED)
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
