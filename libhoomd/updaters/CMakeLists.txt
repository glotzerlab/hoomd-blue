# Maintainer: joaander

# find all header files in this directory
file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# install them
install(FILES ${HDRS} DESTINATION ${INC_INSTALL_DIR})