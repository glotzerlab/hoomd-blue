file(GLOB _SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB _CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

list(APPEND PLUGIN_SRC ${_SRCS})

if (ENABLE_CUDA)
    CUDA_COMPILE(_CUDA_GENERATED_FILES ${_CUDA_SRCS})
    list(APPEND PLUGIN_SRC ${_CUDA_GENERATED_FILES})
endif (ENABLE_CUDA)

# build the C++ python module
set(LIB_TARGET "_${PROJECT_NAME}")
add_library(${LIB_TARGET} ${PLUGIN_SRC} ${PYMODULE_BUILD})
# fix up the file name so it will work as a python module
set_target_properties(${LIB_TARGET} PROPERTIES PREFIX "" OUTPUT_NAME "${LIB_TARGET}")
if(APPLE)
    set_target_properties(${LIB_TARGET} PROPERTIES SUFFIX ".so")
endif(APPLE)

# install the library to the python-module directory (NOTE: maybe we should make a subdirectory to hold all the plugins
# ? it could be hoomd_plugins. Doing so would keep things a little more organized on user's machines
install(TARGETS ${LIB_TARGET}
        LIBRARY DESTINATION lib/hoomd/python-module
        )

