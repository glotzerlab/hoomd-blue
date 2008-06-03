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

find_package(Doxygen)

if (DOXYGEN)

# configure exclusions based on USE_* options
set(DOXYGEN_PREDEFINED "")
set(DOXYGEN_ADDITIONAL_EXCLUSIONS "")
if(USE_CUDA)
	set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED} " USE_CUDA=1")
else(USE_CUDA)
	set(DOXYGEN_ADDITIONAL_EXCLUSIONS " */cuda/* */computes_gpu/*")
endif(USE_CUDA)

if(USE_PYTHON)
	set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED} " \\\nUSE_PYTHON=1")
else(USE_PYTHON)
	set(DOXYGEN_ADDITIONAL_EXCLUSIONS "${DOXYGEN_ADDITIONAL_EXCLUSIONS} */python/* */lib/python-module/*")
endif(USE_PYTHON)

# disable DOT in doxygen if dot is not found
if(DOXYGEN_DOT_PATH)
	set(DOXYGEN_HAVE_DOT "YES")
else(DOXYGEN_DOT_PATH)
	set(DOXYGEN_HAVE_DOT "NO")
endif(DOXYGEN_DOT_PATH)

# setup version number to be included in the documentation
if (HOOMD_SUBVERSION_BUILD)
	set(DOXYGEN_HOOMD_VERSION "svnversion:${SVNVERSION}")
else (HOOMD_SUBVERSION_BUILD)
	set(DOXYGEN_HOOMD_VERSION ${HOOMD_VERSION})
endif (HOOMD_SUBVERSION_BUILD)

# custom html headers, footers and stylesheets have been created to make the
# doxygen docs fit in with the main HOOMD homepage. Enable them as an option
option(USE_DOXYGEN_HOOMD_STYLE OFF "Enable the usage of the HOOMD homepage
stylesheet in processing the doxygen html")
if (USE_DOXYGEN_HOOMD_STYLE)
	set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/header.html)
	set(DOXYGEN_HTML_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/footer.html)
	set(DOXYGEN_HTML_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/stylesheet.css)
else (USE_DOXYGEN_HOOMD_STYLE)
	set(DOXYGEN_HTML_HEADER "")
	set(DOXYGEN_HTML_FOOTER "")
	set(DOXYGEN_HTML_STYLESHEET "")
endif (USE_DOXYGEN_HOOMD_STYLE)

# configure the doxygen file
configure_file (${HOOMD_SOURCE_DIR}/doc/Doxyfile.in ${HOOMD_BINARY_DIR}/doc/Doxyfile)

add_custom_command (OUTPUT ${HOOMD_BINARY_DIR}/doc/html/index.html
					COMMAND ${DOXYGEN} ${HOOMD_BINARY_DIR}/doc/Doxyfile
					DEPENDS ${HOOMD_CPU_SRCS} ${HOOMD_GPU_SRCS}
					${HOOMD_SOURCE_DIR}/cuda/*.cu
					${HOOMD_SOURCE_DIR}/cuda/*.h
					${HOOMD_SOURCE_DIR}/doc/*.doc
					${HOOMD_SOURCE_DIR}/unit_tests/*.cc
					${HOOMD_SOURCE_DIR}/benchmarks/*.cc
					${HOOMD_SOURCE_DIR}/lib/python-module/hoomd_script/*.py
					${HOOMD_SOURCE_DIR}/doc/Doxyfile.in)

add_custom_target (doc ALL DEPENDS ${HOOMD_BINARY_DIR}/doc/html/index.html)

endif (DOXYGEN)
