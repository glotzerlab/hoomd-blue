# $Id$
# $URL$
# Maintainer: joaander

# install the hoomd_script python module
install(DIRECTORY hoomd_plugins hoomd_script
    DESTINATION ${LIB_INSTALL_DIR}/python-module
    PATTERN ".svn" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
    )

# create a convenience target for editing .py files in IDEs
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script/*.py)
add_custom_target(hoomd_script SOURCES ${SRCS})
