# Maintainer: joaander

if (DOXYGEN)

# policy change: always build all documentation even on platforms that don't
# have CUDA
set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED} " \"ENABLE_CUDA=1\" \\
    \"NVCC=1\" \\
    \"BOOST_VERSION=103500\" ")

# build MPI documentation
set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED} " \"ENABLE_MPI=1\" ")

# disable DOT in doxygen if dot is not found
if(DOXYGEN_DOT_PATH)
    set(DOXYGEN_HAVE_DOT "YES")
else(DOXYGEN_DOT_PATH)
    set(DOXYGEN_HAVE_DOT "NO")
endif(DOXYGEN_DOT_PATH)

# setup version number to be included in the documentation
set(DOXYGEN_HOOMD_VERSION ${HOOMD_VERSION_LONG})

# generate a list of source files to depend on
file(GLOB_RECURSE _source_files
                  ${HOOMD_SOURCE_DIR}/libhoomd/*.h
                  ${HOOMD_SOURCE_DIR}/libhoomd/*.cuh
                  ${HOOMD_SOURCE_DIR}/libhoomd/*.cc
                  ${HOOMD_SOURCE_DIR}/libhoomd/*.cu)

file(GLOB_RECURSE _dev_doc_sources
                    ${CMAKE_CURRENT_SOURCE_DIR}/user/*.dox
                    ${CMAKE_CURRENT_SOURCE_DIR}/dev/*.doc
                    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_dev.in)

# configure the doxygen file and target for the developer documentation
set(DOXYGEN_DEV_DOC_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/hoomd-devdoc-${HOOMD_VERSION}")
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_dev.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_dev)
add_custom_command (OUTPUT ${DOXYGEN_DEV_DOC_HTML_DIR}/index.html
                    COMMAND ${DOXYGEN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_dev
                    DEPENDS ${_source_files} ${_dev_doc_sources} ${_gle_images})

add_custom_target (dev ALL DEPENDS ${DOXYGEN_DEV_DOC_HTML_DIR}/index.html
                           SOURCES ${_dev_doc_sources})

# configure the doxygen file and target for the user documentation

file(GLOB_RECURSE _module_files
                  ${HOOMD_SOURCE_DIR}/python-module/*.py)

# custom html headers, footers and stylesheets have been created to make the
# doxygen docs fit in with the main HOOMD homepage. Enable them as an option
option(ENABLE_DOXYGEN_HOOMD_STYLE OFF "Enable the usage of the HOOMD homepage
stylesheet in processing the doxygen html")
mark_as_advanced(ENABLE_DOXYGEN_HOOMD_STYLE)
if (ENABLE_DOXYGEN_HOOMD_STYLE)
    set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/user/header.html)
    set(DOXYGEN_HTML_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/user/footer.html)
    set(DOXYGEN_HTML_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/user/stylesheet.css)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_DISABLE_INDEX YES)
else (ENABLE_DOXYGEN_HOOMD_STYLE)
    set(DOXYGEN_HTML_HEADER "")
    set(DOXYGEN_HTML_FOOTER "")
    set(DOXYGEN_HTML_STYLESHEET "")
    set(DOXYGEN_USE_MATHJAX NO)
    set(DOXYGEN_DISABLE_INDEX NO)
endif (ENABLE_DOXYGEN_HOOMD_STYLE)

# configure the doxygen file
set(DOXYGEN_USER_DOC_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/hoomd-userdoc-${HOOMD_VERSION}")
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_user.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_user)

set (_user_doc_sources  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_user.in
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/installation_guide.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/quick_start.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/examples.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/index.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/xml_file_format.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/command_line_options.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/conversion_scripts.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide_mac.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide_linux_ubuntu.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide_linux_centos.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide_linux_fedora.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide_linux_opensuse.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/compile_guide_linux_generic.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/license.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/citing.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/credits.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/stylesheet.css
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/header.html
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/footer.html
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/mpi.dox
                        ${CMAKE_CURRENT_SOURCE_DIR}/user/box.dox
                        )

add_custom_command (OUTPUT ${DOXYGEN_USER_DOC_HTML_DIR}/index.html
                    COMMAND ${DOXYGEN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_user
                    DEPENDS ${_module_files}
                    ${CMAKE_CURRENT_SOURCE_DIR}/user/*.dox
                    ${CMAKE_CURRENT_SOURCE_DIR}/user/*.html
                    ${CMAKE_CURRENT_SOURCE_DIR}/user/*.css
                    ${CMAKE_CURRENT_SOURCE_DIR}/dev/*.doc
                    ${HOOMD_SOURCE_DIR}/share/hoomd/examples/*
                    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_user.in)

add_custom_target (user ALL DEPENDS ${DOXYGEN_USER_DOC_HTML_DIR}/index.html
                            SOURCES ${_user_doc_sources})

endif (DOXYGEN)
