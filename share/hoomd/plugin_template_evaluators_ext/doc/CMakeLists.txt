find_package(Doxygen)
if (DOXYGEN_FOUND)

set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED} " \"ENABLE_CUDA=1\" \\
    \"NVCC=1\" ")

# disable DOT in doxygen if dot is not found
if(DOXYGEN_DOT_PATH)
    set(DOXYGEN_HAVE_DOT "YES")
else(DOXYGEN_DOT_PATH)
    set(DOXYGEN_HAVE_DOT "NO")
endif(DOXYGEN_DOT_PATH)

# configure the doxygen file
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

file(GLOB SOURCE_FILES "${${PROJECT_NAME}_SOURCE_DIR}/cppmodule/*.cu"
                       "${${PROJECT_NAME}_SOURCE_DIR}/cppmodule/*.cuh"
                       "${${PROJECT_NAME}_SOURCE_DIR}/cppmodule/*.cc"
                       "${${PROJECT_NAME}_SOURCE_DIR}/cppmodule/*.h"
                       "${${PROJECT_NAME}_SOURCE_DIR}/pymodule/*.py")

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/index.html
                    COMMAND ${DOXYGEN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                    DEPENDS
                    ${SOURCE_FILES}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                    ${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml)

add_custom_target (${PROJECT_NAME}_doc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/index.html)

endif (DOXYGEN_FOUND)
