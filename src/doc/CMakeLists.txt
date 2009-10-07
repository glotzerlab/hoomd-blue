# $Id$
# $URL$
# Maintainer: joaander

if (DOXYGEN)

# policy change: always build all documentation even on platforms that don't
# have CUDA/Python/FFT/whatever
set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED} " \"ENABLE_CUDA=1\" \\
    \"ENABLE_FFT=1\" \\
    \"ENABLE_FFTW=1\" \\ 
    \"NVCC=1\" \\
    \"BOOST_VERSION=103500\" ")

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

# configure the doxygen file
configure_file (${HOOMD_SOURCE_DIR}/doc/Doxyfile.in ${HOOMD_BINARY_DIR}/doc/Doxyfile)

add_custom_command (OUTPUT ${HOOMD_BINARY_DIR}/doc/html/index.html
                    COMMAND ${DOXYGEN} ${HOOMD_BINARY_DIR}/doc/Doxyfile
                    DEPENDS ${HOOMD_CPU_SRCS} ${HOOMD_GPU_SRCS}
                    ${HOOMD_SOURCE_DIR}/cuda/*.cu
                    ${HOOMD_SOURCE_DIR}/cuda/*.h
                    ${HOOMD_SOURCE_DIR}/cuda/*.cuh
                    ${HOOMD_SOURCE_DIR}/doc/*.doc
                    ${HOOMD_SOURCE_DIR}/user_doc/*.doc
                    ${HOOMD_SOURCE_DIR}/unit_tests/*.cc
                    ${HOOMD_SOURCE_DIR}/lib/python-module/hoomd_script/*.py
                    ${HOOMD_SOURCE_DIR}/doc/Doxyfile.in
                    ${HOOMD_SOURCE_DIR}/doc/DoxygenLayout.xml)

add_custom_target (doc ALL DEPENDS ${HOOMD_BINARY_DIR}/doc/html/index.html)

endif (DOXYGEN)
