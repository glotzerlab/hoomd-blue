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

if (NOT ENABLE_MINIMAL_INSTALL)

## install the share directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/benchmarks
		DESTINATION ${DATA_INSTALL_DIR}
		CONFIGURATIONS Release
		PATTERN "*.hoomd"
		PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE
		PATTERN ".svn" EXCLUDE
		)
		
## install the icon, mime type and application menu entry in linux
if (NOT WIN32 AND NOT APPLE)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/icons ${CMAKE_CURRENT_SOURCE_DIR}/mime ${CMAKE_CURRENT_SOURCE_DIR}/applications
		DESTINATION share
		CONFIGURATIONS Release
		PATTERN ".svn" EXCLUDE
		)
endif (NOT WIN32 AND NOT APPLE)	
		
## install the icon file in windows
if (WIN32)
   install(FILES ${HOOMD_SOURCE_DIR}/pics_icons/HOOMDScriptIcon.ico
		DESTINATION ${DATA_INSTALL_DIR}
		CONFIGURATIONS Release
		)
endif (WIN32)

endif (NOT ENABLE_MINIMAL_INSTALL)

## install the license file
install(FILES ${HOOMD_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/hoomd_data_dir
		DESTINATION ${DATA_INSTALL_DIR}
		CONFIGURATIONS Release
		)
