# $Id$
# $URL$
# Maintainer: joaander

if (NOT ENABLE_MINIMAL_INSTALL)

## install the share directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/benchmarks
        DESTINATION ${DATA_INSTALL_DIR}
        CONFIGURATIONS Release
        PATTERN "*.hoomd"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE
        PATTERN ".svn" EXCLUDE
        )
        
## install the icon, mime type and application menu entry in linux
if (NOT WIN32 AND NOT APPLE)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/icons ${CMAKE_CURRENT_SOURCE_DIR}/mime ${CMAKE_CURRENT_SOURCE_DIR}/applications
        DESTINATION share
        CONFIGURATIONS Release
        PATTERN ".svn" EXCLUDE
        )
endif (NOT WIN32 AND NOT APPLE)	
        
## install the icon file in windows
if (WIN32)
install(FILES ${HOOMD_SOURCE_DIR}/pics_icons/HOOMDScriptIcon.ico
        DESTINATION ${DATA_INSTALL_DIR}
        CONFIGURATIONS Release
        )
endif (WIN32)

endif (NOT ENABLE_MINIMAL_INSTALL)

## install the license file
install(FILES ${HOOMD_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/hoomd_data_dir
        DESTINATION ${DATA_INSTALL_DIR}
        CONFIGURATIONS Release
        )
