# $Id$
# $URL$
# Maintainer: joaander

# install the hoomd_script python module
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${LIB_INSTALL_DIR}/python-module
    PATTERN ".svn" EXCLUDE
    )
