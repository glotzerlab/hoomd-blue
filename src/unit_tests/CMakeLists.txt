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

######
# setup libraries to link
# only add the boost library if not win32, since it provides an auto-linking mechanism
if (WIN32)
	set(LIBS ${HOOMD_LIB} ${WINSOCK_LIB})
else (WIN32)
	set(LIBS ${HOOMD_LIB} 
			${Boost_THREAD_LIBRARY}
			${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
			${Boost_FILESYSTEM_LIBRARY}
			${Boost_SIGNALS_LIBRARY}
			${CMAKE_THREAD_LIBS_INIT})

	if(Boost_SYSTEM_LIBRARY)
		set(LIBS ${LIBS} ${Boost_SYSTEM_LIBRARY})
	endif(Boost_SYSTEM_LIBRARY)

endif (WIN32)

####################################
## Setup valgrind options
find_program(VALGRIND valgrind DOC 
	"valgrind executable for advanced regression tests")

## User gets an option to enable valgrind tests
if (VALGRIND)
	option(USE_VALGRIND OFF "Enable running of unit tests through valgrind")

	set(VALGRIND_COMMON_ARGS "--leak-check=full;--suppressions=${CMAKE_CURRENT_SOURCE_DIR}/cudart_valgrind.supp;--error-exitcode=1")
endif (VALGRIND)

enable_testing()

###################################
## Setup all of the test executables in a for loop
set(TEST_LIST utils_test pdata_test neighborlist_test bond_force_test fenebond_force_test
	lj_force_test lj_wall_compute_test nve_updater_test nvt_updater_test npt_updater_test 
        temp_rescale_updater_test xml_reader_writer_test system_test ElectrostaticShortRange_force_test)

if(USE_CUDA)
	set(TEST_LIST ${TEST_LIST} gpu_worker_test)
endif(USE_CUDA)

foreach (CUR_TEST ${TEST_LIST})
	# add and link the unit test executable
	add_executable(${CUR_TEST} ${CUR_TEST}.cc)
	target_link_libraries(${CUR_TEST} ${LIBS})

	# add it to the unit test list
	get_target_property(CUR_TEST_EXE ${CUR_TEST} LOCATION)

	# unfortunately, this needs to be handled differently in visual studio
	if (MSVC)
		add_test(${CUR_TEST} ${EXECUTABLE_OUTPUT_PATH}/${CUR_TEST})
	else (MSVC)
		add_test(${CUR_TEST} ${CUR_TEST_EXE})
		# add a valgrind executed test if requested
		if (USE_VALGRIND)
			add_test(valgrind_${CUR_TEST} ${VALGRIND} ${VALGRIND_COMMON_ARGS} ${CUR_TEST_EXE})
		endif (USE_VALGRIND)
	endif (MSVC)
endforeach (CUR_TEST)
