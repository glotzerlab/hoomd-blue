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
	set(CPACK_NSIS_URL_INFO_ABOUT "http://www.ameslab.gov/hoomd/")
	set(CPACK_NSIS_CONTACT "joaander@iastate.edu")
	set(CPACK_NSIS_DISPLAY_NAME "HOOMD")
	
	set(CPACK_NSIS_MODIFY_PATH ON)
	set(CPACK_PACKAGE_EXECUTABLES "hoomd;HOOMD Python Interpreter")
	
	# register .hoomd as a file type
	# the cool massive quadruple backspaces are to get two backspaces after the escape and then
	# ANOTHER two to be escaped again at the next stage of processing!
	# (why in the world does winblows need to use \ as a path separator...)
	
	# The other confusing thing is the triple back spaces \" escapes the quote at this state
	# in the processing. Then the \\ will turn into a \, so the next stage of processing sees
	# \" to turn into " for the final file. Whew!
	
	set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
	WriteRegStr HKCR \\\".hoomd\\\" \\\"\\\" \\\"HOOMD.Script\\\"
	WriteRegStr HKCR \\\"HOOMD.Script\\\" \\\"\\\" \\\"HOOMD Script File\\\"
    WriteRegStr HKCR \\\"HOOMD.Script\\\\shell\\\" \\\"\\\" \\\"open\\\"
    WriteRegStr HKCR \\\"HOOMD.Script\\\\shell\\\\open\\\\command\\\" \\\"\\\" '\\\"$INSTDIR\\\\bin\\\\hoomd.exe\\\" \\\"%1\\\"'
    !define SHCNE_ASSOCCHANGED 0x8000000
	!define SHCNF_IDLIST 0
	System::Call 'Shell32::SHChangeNotify(i ${SHCNE_ASSOCCHANGED}, i ${SHCNF_IDLIST}, i 0, i 0)'"
	)
	
	# setup the uninstall for the .hoomd file type
	set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
	DeleteRegKey HKCR \\\".hoomd\\\"
	DeleteRegKey HKCR \\\"HOOMD.Script\\\"
	System::Call 'Shell32::SHChangeNotify(i ${SHCNE_ASSOCCHANGED}, i ${SHCNF_IDLIST}, i 0, i 0)'"
	)
	
else (WIN32 AND NOT UNIX)
	set(CPACK_STRIP_FILES ${LIB_INSTALL_DIR}/hoomd ${LIB_INSTALL_DIR}/force_compute_bmark ${LIB_INSTALL_DIR}/neighborlist_bmark ${LIB_INSTALL_DIR}/pdata_gpucopy_bmark)
else (WIN32 AND NOT UNIX)
	# on linux, I want to have x86_64 or x86 in the name of the package
	find_program(UNAME_EXE "uname" DOC "uname executable")
	mark_as_advanced(UNAME_EXE)
	if (UNAME_EXE)
		exec_program(${UNAME_EXE} ${HOOMD_SOURCE_DIR} ARGS -m OUTPUT_VARIABLE UNAME_MACHINE)
	else (UNAME_EXE)
		set(UNAME_MACHINE "NA")
	endif (UNAME_EXE)
	
	set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${UNAME_MACHINE}")
endif (WIN32 AND NOT UNIX)	

if (WIN32)
	set (CPACK_GENERATOR "NSIS")
else (WIN32)
	set (CPACK_GENERATOR "TBZ2")
endif (WIN32)

# use our overrided NSIS template files
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
include(CPack)