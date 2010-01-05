## $Id$
## $URL$
# Maintainer: joaander

if (ENABLE_CUDA)

# find all source files in this directory
file(GLOB CU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

# add them to the source group
source_group(cuda FILES ${CU_SRCS} ${HDRS})

# add them to the main libhoomd list
set(_libhoomd_sources ${_libhoomd_sources} ${HDRS} PARENT_SCOPE)
set(_libhoomd_cu_sources ${_libhoomd_cu_sources} ${CU_SRCS} PARENT_SCOPE)

# install headers
install(FILES ${HDRS} DESTINATION ${INC_INSTALL_DIR})

endif(ENABLE_CUDA)

