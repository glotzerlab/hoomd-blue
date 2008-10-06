
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

# enable unit tests
enable_testing()

# setup the library compile mode
if(USE_STATIC)
	set(LIB_MODE STATIC)
	add_definitions(-DBOOST_PYTHON_STATIC_LIB)
else(USE_STATIC)
	set(LIB_MODE SHARED)
endif(USE_STATIC)

# find the python libraries to link to
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# here is where we actually compile the python module version of the library
if (USE_CUDA)
cuda_add_library (${HOOMD_PYTHON_MODULE} ${LIB_MODE} ${HOOMD_CPU_SRCS}
${HOOMD_GPU_SRCS} ${PYTHON_SRCS})
else (USE_CUDA)
add_library (${HOOMD_PYTHON_MODULE} ${LIB_MODE} ${HOOMD_CPU_SRCS}
${PYTHON_SRCS})
endif (USE_CUDA)

set(BOOST_LIBS ${Boost_THREAD_LIBRARY} 
			   ${Boost_FILESYSTEM_LIBRARY} ${Boost_PYTHON_LIBRARY}
			   ${Boost_SIGNALS_LIBRARY})

if (Boost_SYSTEM_LIBRARY)
	set(BOOST_LIBS ${BOOST_LIBS} ${Boost_SYSTEM_LIBRARY})
endif (Boost_SYSTEM_LIBRARY)

##################################
# change the name of the library if it is dynamic
if (NOT USE_STATIC)
	# link the library to its dependancies (but only on non win32 since the linking
	# is automatic there
	if(NOT WIN32)
		target_link_libraries(${HOOMD_PYTHON_MODULE} ${BOOST_LIBS} ${PYTHON_LIBRARIES} ${FFTW_LIBRARAY})
	endif(NOT WIN32)

	if(WIN32)
		target_link_libraries(${HOOMD_PYTHON_MODULE} ${WINSOCK_LIB} ${FFTW_LIBRARAY})
	endif(WIN32)
	# tweak the properties of the output to make a more useful python module
	set_target_properties(${HOOMD_PYTHON_MODULE} PROPERTIES PREFIX "" OUTPUT_NAME "hoomd")

	# .dylib is not regognized as a python module by python on Mac OS X
	if(APPLE)
		set_target_properties(${HOOMD_PYTHON_MODULE} PROPERTIES SUFFIX ".so")
	endif(APPLE)
	
	# also: only install the library if it is dynamic  and not in windows
	if (NOT WIN32)
		install(TARGETS ${HOOMD_PYTHON_MODULE}
			LIBRARY DESTINATION ${LIB_INSTALL_DIR}/python-module
			CONFIGURATIONS Release
			)
			
		# installs the hoomd runner script
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd_dynamic.in 
						${CMAKE_CURRENT_BINARY_DIR}/hoomd_dynamic 
						@ONLY
						)
		install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_dynamic
			DESTINATION ${BIN_INSTALL_DIR}
			CONFIGURATIONS Release
			RENAME hoomd
			)
	endif (NOT WIN32)
endif (NOT USE_STATIC)	

########################
## now setup the hoomd executable

## An update to to CentOS5's python broke linking of the hoomd exe. According
## to an ancient post online, adding -lutil fixed this in python 2.2
set(ADDITIONAL_LIBS "")
if (UNIX AND NOT APPLE)
	find_library(UTIL_LIB util /usr/lib)
	find_library(DL_LIB dl /usr/lib)
	set(ADDITIONAL_LIBS ${UTIL_LIB} ${DL_LIB})
endif (UNIX AND NOT APPLE)
	
# setup the libraries to link to
# only add the boost library if not win32, since it provides an auto-linking mechanism
if (WIN32)
    set(LIBS ${HOOMD_PYTHON_MODULE} ${WINSOCK_LIB} ${PYTHON_LIBRARIES} ${FFTW_LIBRARAY})
else (WIN32)
    set(LIBS ${HOOMD_PYTHON_MODULE} ${BOOST_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${PYTHON_LIBRARIES} ${FFTW_LIBRARAY} ${ADDITIONAL_LIBS})
endif (WIN32)

# setup the python runner executable
# it won't link dynamically with the changed name above, but there really isn't a need to create a dynamically
# linked version as it can be loaded into python directly
if (USE_STATIC)
	add_executable(${HOOMD_EXE} hoomd.cc)
	target_link_libraries(${HOOMD_EXE} ${LIBS})
	set_target_properties(${HOOMD_EXE} 
		PROPERTIES PREFIX "" 
		OUTPUT_NAME "hoomd"
		)
	
	# install the executable
	install(TARGETS ${HOOMD_EXE}
		RUNTIME DESTINATION ${LIB_INSTALL_DIR}
		CONFIGURATIONS Release
		)
		
	# installs the hoomd runner script
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd_static.in 
					${CMAKE_CURRENT_BINARY_DIR}/hoomd_static
					@ONLY
					)	
	install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_static
			DESTINATION ${BIN_INSTALL_DIR}
			CONFIGURATIONS Release
			RENAME hoomd
			)
			
	# add the hoomd_script tests to the unit test list
	get_target_property(CUR_TEST_EXE ${HOOMD_EXE} LOCATION)

	# unfortunately, this needs to be handled differently in visual studio
	if (MSVC)
		add_test(hoomd_script_tests_cpu ${EXECUTABLE_OUTPUT_PATH}/hoomd ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=cpu)
		if (USE_CUDA)
			add_test(hoomd_script_tests_gpu ${EXECUTABLE_OUTPUT_PATH}/hoomd ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=gpu)
		endif (USE_CUDA)
	else (MSVC)
		add_test(hoomd_script_tests_cpu ${CUR_TEST_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=cpu)
		
		if (USE_CUDA)
			add_test(hoomd_script_tests_gpu ${CUR_TEST_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=gpu)
		endif (USE_CUDA)
	endif (MSVC)
	
endif (USE_STATIC)

#################################
## Create source file groups
source_group(utils FILES ${UTILS_SRCS})
source_group(data_structures FILES ${DATA_STRUCT_SRCS})
source_group(computes FILES ${COMPUTES_SRCS})
source_group(computes_gpu FILES ${COMPUTES_GPU_SRCS})
source_group(analyzers FILES ${ANALYZER_SRCS})
source_group(extern FILES ${EXTERN_SRCS})
source_group(updaters FILES ${UPDATER_SRCS})
source_group(updaters_gpu FILES ${UPDATER_GPU_SRCS})
source_group(cuda FILES ${CUDA_SRCS})
source_group(system FILES ${SYSTEM_SRCS})
source_group(python FILES ${PYTHON_SRCS})
