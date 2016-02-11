# Maintainer: joaander

## install the icon, mime type and application menu entry in linux
if (NOT WIN32 AND NOT APPLE)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/icons ${CMAKE_CURRENT_SOURCE_DIR}/mime ${CMAKE_CURRENT_SOURCE_DIR}/applications
        DESTINATION share
        PATTERN ".svn" EXCLUDE
        )
endif (NOT WIN32 AND NOT APPLE)

## install the license file
install(FILES ${HOOMD_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/hoomd_data_dir
        DESTINATION ${DATA_INSTALL_DIR}
        )

## install the plugin templates
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/plugin_template_cpp
        DESTINATION ${DATA_INSTALL_DIR}
        PATTERN ".svn" EXCLUDE
        )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/plugin_template_evaluators_ext
        DESTINATION ${DATA_INSTALL_DIR}
        PATTERN ".svn" EXCLUDE
        )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/examples
        DESTINATION ${DATA_INSTALL_DIR}
        PATTERN ".svn" EXCLUDE
        )
