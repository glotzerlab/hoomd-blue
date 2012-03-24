# Maintainer: joaander

# find all header files in this directory
file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

# install them
install(FILES ${HDRS} DESTINATION ${INC_INSTALL_DIR})

# configure the version info header
configure_file (${HOOMD_SOURCE_DIR}/libhoomd/utils/HOOMDVersion.h.in ${HOOMD_BINARY_DIR}/include/HOOMDVersion.h)

# and install it
install(FILES ${HOOMD_BINARY_DIR}/include/HOOMDVersion.h DESTINATION ${INC_INSTALL_DIR})
