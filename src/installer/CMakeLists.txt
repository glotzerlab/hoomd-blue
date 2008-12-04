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

# $Id$
# $URL$

if (NOT I_WILL_NOT_DISTRUBUTE_THIS_HOOMD)

set(CPACK_DESCRIPTION_SUMMARY "Highly Optimized Object Oriented Molecular Dynamics")
set(CPACK_PACKAGE_VENDOR "Ames Laboratory Iowa State University")
set(CPACK_PACKAGE_VERSION_MAJOR ${HOOMD_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${HOOMD_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${HOOMD_VERSION_PATCH})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "hoomd")

# the package name is upper case now, force it lower here to be pretty
set(CPACK_PACKAGE_NAME "hoomd")

set(CPACK_RESOURCE_FILE_LICENSE "${HOOMD_SOURCE_DIR}/LICENSE")

if (WIN32 AND NOT UNIX)
	set(CPACK_NSIS_INSTALLER_ICON "${HOOMD_SOURCE_DIR}/pics_icons/InstallIcon.ico")
	set(CPACK_NSIS_URL_INFO_ABOUT "http://www.ameslab.gov/hoomd/")
	set(CPACK_NSIS_CONTACT "joaander@iastate.edu")
	set(CPACK_NSIS_DISPLAY_NAME "HOOMD")
		
	set(CPACK_NSIS_MODIFY_PATH ON)
	set(CPACK_PACKAGE_EXECUTABLES "hoomd;HOOMD Python Interpreter")
	
else (WIN32 AND NOT UNIX)
	set(CPACK_STRIP_FILES ${LIB_INSTALL_DIR}/hoomd ${LIB_INSTALL_DIR}/force_compute_bmark ${LIB_INSTALL_DIR}/neighborlist_bmark ${LIB_INSTALL_DIR}/pdata_gpucopy_bmark)

	# on linux, I want to have x86_64 or x86 in the name of the package
	find_program(UNAME_EXE "uname" DOC "uname executable")
	mark_as_advanced(UNAME_EXE)
	if (UNAME_EXE)
		exec_program(${UNAME_EXE} ${HOOMD_SOURCE_DIR} ARGS -m OUTPUT_VARIABLE UNAME_MACHINE)
	else (UNAME_EXE)
		set(UNAME_MACHINE "NA")
	endif (UNAME_EXE)
	
	set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${UNAME_MACHINE}")

	# I also want to append the Python version. This method is potentially
	# faulty as it is not necesarily true that the found python interpreter is
	# the same as the one that is linked against. But then, packaging will
	# only be done by me and I know what I'm doing with this.
	find_package(PythonInterp REQUIRED)
	exec_program(${PYTHON_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS -V
			OUTPUT_VARIABLE PYTHON_VER_UNFORMATTED)
	# remove the space from the string
	string(REPLACE " " "" PYTHON_VER_NOSPACE ${PYTHON_VER_UNFORMATTED})
	# remove the minor version
	string(SUBSTRING ${PYTHON_VER_NOSPACE} 0 9 PYTHON_VER)

	set(CPACK_SYSTEM_NAME "${CPACK_SYSTEM_NAME}-${PYTHON_VER}")
endif (WIN32 AND NOT UNIX)	

if (WIN32)
	set (CPACK_GENERATOR "NSIS")
else (WIN32)
	set (CPACK_GENERATOR "TBZ2")
endif (WIN32)

# use our overrided NSIS template files
# different directories are needed for 64 bit builds vs 32 bit ones

if (CMAKE_CL_64)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/x86_64" ${CMAKE_MODULE_PATH})
else (CMAKE_CL_64)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/i386" ${CMAKE_MODULE_PATH})
endif (CMAKE_CL_64)

include(CPack)

# need to install vs dlls on windows
if(WIN32)
include(InstallRequiredSystemLibraries)

# we don't get any feedback if they were installed: check if the redist dir was found
if (MSVC80)
if (NOT MSVC80_REDIST_DIR)
message("MSVC80 redist dir not found, dlls needed to run HOOMD will not be 
included in the installer. You can safely ignore this message if you do not intend to
build windows installers for HOOMD on this machine.")
endif (NOT MSVC80_REDIST_DIR)
elseif (MSVC90)
if (NOT MSVC90_REDIST_DIR)
message("MSVC90 redist dir not found, dlls needed to run HOOMD will not be 
included in the installer. You can safely ignore this message if you do not intend to
build windows installers for HOOMD on this machine.")
endif (NOT MSVC90_REDIST_DIR)
endif(MSVC80)

endif(WIN32)

endif (NOT I_WILL_NOT_DISTRUBUTE_THIS_HOOMD)
