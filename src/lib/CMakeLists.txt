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

if(USE_STATIC)
	set(LIB_MODE STATIC)
else(USE_STATIC)
	set(LIB_MODE SHARED)
endif(USE_STATIC)

# here is where we actually compile the library
if (USE_CUDA)
cuda_add_library (${HOOMD_LIB} ${LIB_MODE} ${HOOMD_CPU_SRCS} ${HOOMD_GPU_SRCS})
else (USE_CUDA)
add_library (${HOOMD_LIB} ${LIB_MODE} ${HOOMD_CPU_SRCS})
endif (USE_CUDA)

# link the library to its dependancies (but only on non win32 since the linking
# is automatic there
if(NOT WIN32)
	target_link_libraries(${HOOMD_LIB} ${BOOST_THREAD_LIBRARY} ${BOOST_SIGNALS_LIBRARY})
endif(NOT WIN32)

if(WIN32)
	target_link_libraries(${HOOMD_LIB} ${WINSOCK_LIB})
endif(WIN32)

# install the library if this is not a static build
if (NOT USE_STATIC)
	install(TARGETS ${HOOMD_LIB}
		LIBRARY DESTINATION lib
		CONFIGURATIONS Release
		)
endif (NOT USE_STATIC)

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
