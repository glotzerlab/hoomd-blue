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
# $ $URL$

###############################
# UNIX like OSes need a few settings for the unit test library
if (NOT WIN32)
	# first, they need the boost unit test framework added as a library
	set(ADDITIONAL_LIBS hoomd_python_module
			${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
			)

	# then we need to specify BOOST_TEST_DYN_LINK if the unit test framework library is a shared lib
	if("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.dylib$")
	add_definitions(-DBOOST_TEST_DYN_LINK)
	endif("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.dylib$")
	
	if("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")
	add_definitions(-DBOOST_TEST_DYN_LINK)
	endif("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")
endif (NOT WIN32)

####################################
## Setup valgrind options
if (UNIX AND NOT APPLE)
find_program(VALGRIND valgrind DOC 
	"valgrind executable for advanced regression tests")
endif (UNIX AND NOT APPLE)

## User gets an option to enable valgrind tests
if (VALGRIND)
	option(ENABLE_VALGRIND OFF "Enable running of unit tests through valgrind")

	set(VALGRIND_COMMON_ARGS "--leak-check=full;--suppressions=${CMAKE_CURRENT_SOURCE_DIR}/cudart_valgrind.supp;--error-exitcode=1")
	mark_as_advanced(VALGRIND)
endif (VALGRIND)

enable_testing()

###################################
## Setup all of the test executables in a for loop
set(TEST_LIST
	gpu_array_test
	pdata_test
	particle_group_test
	utils_test
	harmonic_bond_force_test
	fenebond_force_test
	lj_wall_compute_test
	neighborlist_test
	lj_force_test
	shiftedlj_force_test
	gaussian_force_test
	nve_updater_test
	nvt_updater_test
	zero_momentum_updater_test
	temp_rescale_updater_test
	xml_reader_writer_test system_test
	ElectrostaticShortRange_force_test)
	
if (ENABLE_FFT)
	set(TEST_LIST ${TEST_LIST} 
	ElectrostaticLongRange_PPPM_test
	Fourier_transform_test)
endif (ENABLE_FFT)

if(ENABLE_CUDA)
	set(TEST_LIST ${TEST_LIST} 
	gpu_worker_test)
endif(ENABLE_CUDA)

	# put the longest tests last
	set(TEST_LIST ${TEST_LIST}
	bdnvt_updater_test
	npt_updater_test)

foreach (CUR_TEST ${TEST_LIST})
	# add and link the unit test executable
	if(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
	cuda_add_executable(${CUR_TEST} ${CUR_TEST}.cc ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
	else(ENABLEC_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
	add_executable(${CUR_TEST} ${CUR_TEST}.cc)
	endif(ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUR_TEST}.cu)
	
	target_link_libraries(${CUR_TEST} hoomd_python_module ${ADDITIONAL_LIBS} ${HOOMD_COMMON_LIBS})

	# add it to the unit test list
	get_target_property(CUR_TEST_EXE ${CUR_TEST} LOCATION)

	# unfortunately, this needs to be handled differently in visual studio
	if (MSVC)
		add_test(${CUR_TEST} ${EXECUTABLE_OUTPUT_PATH}/${CUR_TEST})
	else (MSVC)
		add_test(${CUR_TEST} ${CUR_TEST_EXE})
		# add a valgrind executed test if requested
		if (ENABLE_VALGRIND)
			add_test(valgrind_${CUR_TEST} ${VALGRIND} ${VALGRIND_COMMON_ARGS} ${CUR_TEST_EXE})
		endif (ENABLE_VALGRIND)
	endif (MSVC)
endforeach (CUR_TEST)
