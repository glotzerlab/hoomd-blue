# $Id$
# $URL$
# Maintainer: joaander

if (NOT ENABLE_MINIMAL_INSTALL)

## install the file conversion scripts
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/hoomd2lammps_input.py ${CMAKE_CURRENT_SOURCE_DIR}/hoomd2lammps_dump.py
        DESTINATION ${BIN_INSTALL_DIR}
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE
        CONFIGURATIONS Release
        )

endif (NOT ENABLE_MINIMAL_INSTALL)
