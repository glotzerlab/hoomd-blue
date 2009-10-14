## install the python module
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION lib/hoomd/python-module/hoomd_plugins/${PROJECT_NAME}
        PATTERN ".svn" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        )

