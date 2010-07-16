## install the python module
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION hoomd_plugins/${PROJECT_NAME}
        PATTERN ".svn" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        )

# create a convenience target for editing .py files in IDEs
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
add_custom_target (${PROJECT_NAME}_py SOURCES ${SRCS})

