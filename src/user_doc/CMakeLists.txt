# $Id: CMakeLists.txt 1029 2008-06-03 20:30:06Z joaander $
# $URL: http://svn2.assembla.com/svn/hoomd/branches/gpu-iface-rewrite/src/doc/CMakeLists.txt $
# Maintainer: joaander

if (DOXYGEN)

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
option(ENABLE_DOXYGEN_HOOMD_STYLE OFF "Enable the usage of the HOOMD homepage
stylesheet in processing the doxygen html")
mark_as_advanced(ENABLE_DOXYGEN_HOOMD_STYLE)
if (ENABLE_DOXYGEN_HOOMD_STYLE)
    set(DOXYGEN_HTML_HEADER ${HOOMD_SOURCE_DIR}/user_doc/header.html)
    set(DOXYGEN_HTML_FOOTER ${HOOMD_SOURCE_DIR}/user_doc/footer.html)
    set(DOXYGEN_HTML_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/stylesheet.css)
    set(DOXYGEN_DISABLE_INDEX YES)
else (ENABLE_DOXYGEN_HOOMD_STYLE)
    set(DOXYGEN_HTML_HEADER "")
    set(DOXYGEN_HTML_FOOTER "")
    set(DOXYGEN_HTML_STYLESHEET "")
    set(DOXYGEN_DISABLE_INDEX NO)
endif (ENABLE_DOXYGEN_HOOMD_STYLE)

# configure the doxygen file
configure_file (${HOOMD_SOURCE_DIR}/user_doc/Doxyfile.in ${HOOMD_BINARY_DIR}/user_doc/Doxyfile)

add_custom_command (OUTPUT ${HOOMD_BINARY_DIR}/user_doc/html/index.html
                    COMMAND ${DOXYGEN} ${HOOMD_BINARY_DIR}/user_doc/Doxyfile
                    DEPENDS ${HOOMD_CPU_SRCS} ${HOOMD_GPU_SRCS}
                    ${HOOMD_SOURCE_DIR}/user_doc/*.doc
                    ${HOOMD_SOURCE_DIR}/user_doc/*.html
                    ${HOOMD_SOURCE_DIR}/user_doc/*.css
                    ${HOOMD_SOURCE_DIR}/doc/*.doc
                    ${HOOMD_SOURCE_DIR}/lib/python-module/hoomd_script/*.py
                    ${HOOMD_SOURCE_DIR}/user_doc/Doxyfile.in
                    ${HOOMD_SOURCE_DIR}/user_doc/DoxygenLayout.xml)

add_custom_target (user_doc ALL DEPENDS ${HOOMD_BINARY_DIR}/user_doc/html/index.html)

endif (DOXYGEN)
