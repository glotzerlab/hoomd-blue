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

# libraries
# only add the boost library if not win32, since it provides an auto-linking mechanism

set(LIBS hoomd_python_module ${HOOMD_COMMON_LIBS})

###################################
## Setup all of the test executables
add_executable(neighborlist_bmark neighborlist_bmark.cc)
target_link_libraries(neighborlist_bmark ${LIBS})

add_executable(force_compute_bmark force_compute_bmark.cc)
target_link_libraries(force_compute_bmark ${LIBS})

# install the executables
install(TARGETS neighborlist_bmark force_compute_bmark
	RUNTIME DESTINATION ${LIB_INSTALL_DIR}
	CONFIGURATIONS Release
	)
	
if (ENABLE_CUDA)
add_executable(pdata_gpucopy_bmark pdata_gpucopy_bmark.cc)
target_link_libraries(pdata_gpucopy_bmark ${LIBS})

# install the executables
install(TARGETS pdata_gpucopy_bmark
	RUNTIME DESTINATION ${LIB_INSTALL_DIR}
	CONFIGURATIONS Release
	)
endif (ENABLE_CUDA)

