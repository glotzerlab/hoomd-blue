# $Id$
# $URL$
# Maintainer: joaander

# install the hoomd_script python module
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python-module
    DESTINATION ${LIB_INSTALL_DIR}
    CONFIGURATIONS Release
    PATTERN ".svn" EXCLUDE
    )
