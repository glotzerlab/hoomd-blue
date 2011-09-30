## $Id$
## $URL$
# Maintainer: joaander

# Specify cmake modules path to use our custom modules
set(CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    "${CMAKE_CURRENT_SOURCE_DIR}/hoomd"
    "${CMAKE_CURRENT_SOURCE_DIR}/git"
    "${CMAKE_MODULE_PATH}"
    PARENT_SCOPE)

## install the helper modules in linux and mac
if (NOT WIN32)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cuda
                  ${CMAKE_CURRENT_SOURCE_DIR}/hoomd
        DESTINATION ${DATA_INSTALL_DIR}/CMake
        PATTERN ".svn" EXCLUDE
        )
endif (NOT WIN32)
