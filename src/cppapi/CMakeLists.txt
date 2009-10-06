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


# $Id: CMakeLists.txt 1893 2009-06-15 12:53:11Z joaander $
# $URL: http://codeblue.umich.edu/hoomd-blue/svn/trunk/src/lib/CMakeLists.txt $

#################################
## write a C++ include file and a config.h file
option(ENABLE_CXX "Create headers and library variables to link c++ codes to the hoomd library" OFF)
mark_as_advanced(ENABLE_CXX)
if (ENABLE_CXX)
# set up a config.h file to include the appropriate precompiler directives
configure_file(${HOOMD_SOURCE_DIR}/cxx/Config.h.cmake ${HOOMD_BINARY_DIR}/cxx/config.h)
## set up a "hoomd.h" header file that includes the "config.h" file as well as the other headers 
file(WRITE ${HOOMD_BINARY_DIR}/cxx/hoomd.h "#ifndef _HOOMD_H\n#define _HOOMD_H\n\n")
file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd.h "#include \"config.h\"\n") 
file(GLOB HEADER_FILES "${HOOMD_SOURCE_DIR}/*/*.h")
# strip the path
# string(REPLACE "${HOOMD_SOURCE_DIR}/" "" HEADER_FILES "${HEADER_FILES}")
# replace double slash with single slash
# string(REPLACE "//" "/" HEADER_FILES "${HEADER_FILES}")
foreach(_FILENAME ${HEADER_FILES})
	get_filename_component(_FILENAMENOPATH ${_FILENAME} NAME)
	string(REGEX MATCH "GPU" _MATCH "${_FILENAME}")
	set (_DONT_SKIP_THIS 0)
	if (NOT ENABLE_CUDA)
		string(COMPARE EQUAL "${_MATCH}" "" _DONT_SKIP_THIS)
	endif (NOT ENABLE_CUDA)
	if (_DONT_SKIP_THIS)
		file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd.h "#include \"${_FILENAMENOPATH}\"\n")
	endif (_DONT_SKIP_THIS)
endforeach(_FILENAME ${HEADER_FILES})
file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd.h "\n#endif\n\n")

# copy the headers into the include/hoomd folder
file(GLOB files "${HOOMD_SOURCE_DIR}/*/*.h")
install(FILES ${files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hoomd)
file(GLOB files "${HOOMD_BINARY_DIR}/cxx/*.h")
install(FILES ${files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hoomd)

## write a list of libraries used for linking.
file(WRITE ${HOOMD_BINARY_DIR}/cxx/hoomd_linker_setup.sh "#!/bin/bash\n\n")
file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd_linker_setup.sh "HOOMD_LINK_LIBS=\" \\\n")
#message(STATUS ${HOOMD_COMMON_LIBS})
set (_SKIP_THIS_AND_NEXT 0)
set (_SKIP_THIS 0)
foreach(_LIBRARY ${HOOMD_COMMON_LIBS})
	if (_SKIP_THIS_AND_NEXT)
		set (_SKIP_THIS_AND_NEXT 0)
	else (_SKIP_THIS_AND_NEXT)
		string(COMPARE EQUAL ${_LIBRARY} "debug" _SKIP_THIS_AND_NEXT)
		set (_SKIP_THIS 0)
		string(COMPARE EQUAL ${_LIBRARY} "optimized" _SKIP_THIS)
		if (NOT _SKIP_THIS AND NOT _SKIP_THIS_AND_NEXT)
			file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd_linker_setup.sh "${_LIBRARY} \\\n")
		endif (NOT _SKIP_THIS AND NOT _SKIP_THIS_AND_NEXT)
	endif (_SKIP_THIS_AND_NEXT)
endforeach(_LIBRARY ${HOOMD_COMMON_LIBS})
file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd_linker_setup.sh "${CMAKE_INSTALL_PREFIX}/lib/libhoomd_python_module.a \\\n")
file(APPEND ${HOOMD_BINARY_DIR}/cxx/hoomd_linker_setup.sh "-I${CMAKE_INSTALL_PREFIX}/include/hoomd\"")
endif (ENABLE_CXX)
