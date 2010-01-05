## $Id$
## $URL$
# Maintainer: joaander

# find all source files in this directory
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# add them to the source group
source_group(computes FILES ${SRCS} ${HDRS})

# add them to the main libhoomd list
set(_libhoomd_sources ${_libhoomd_sources} ${SRCS} ${HDRS} PARENT_SCOPE)

# install headers
install(FILES ${HDRS} DESTINATION ${INC_INSTALL_DIR})