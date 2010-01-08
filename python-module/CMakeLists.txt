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
