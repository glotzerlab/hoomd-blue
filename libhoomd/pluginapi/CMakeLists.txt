# $Id$
# $URL$
# Maintainer: joaander

#################################
## setup hoomd_config.h with current configuration options

# set up a hoomd_config.h file to include the appropriate precompiler directives
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/hoomd_config.h)
# install it
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hoomd_config.h DESTINATION ${INC_INSTALL_DIR})
# also install hoomd.h
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/hoomd.h DESTINATION ${INC_INSTALL_DIR})

#################################
## setup hoomd-config.sh script for quick & dirty compiler option setup

# build a list of libraries used for linking.
set (PLUGINAPI_LIBS "")
# the name of the hoomd library for linking changes between static and dynamic builds
get_target_property(_HOOMD_LIB_LOC hoomd_python_module LOCATION)
get_filename_component(_HOOMD_LIB_NAME ${_HOOMD_LIB_LOC} NAME)
if (ENABLE_STATIC)
set (_HOOMD_LIB_RELPATH "lib")
else (ENABLE_STATIC)
set (_HOOMD_LIB_RELPATH "lib/hoomd/python-module")
endif (ENABLE_STATIC)
set(PLUGINAPI_LIBS "${PLUGINAPI_LIBS} \${HOOMD_ROOT}/${_HOOMD_LIB_RELPATH}/${_HOOMD_LIB_NAME} ")

# add all the common libs
set (_SKIP_THIS_AND_NEXT 0)
set (_SKIP_THIS 0)
foreach(_LIBRARY ${HOOMD_COMMON_LIBS})
    if (_SKIP_THIS_AND_NEXT)
        set (_SKIP_THIS_AND_NEXT 0)
    else (_SKIP_THIS_AND_NEXT)
        string(COMPARE EQUAL ${_LIBRARY} "debug" _SKIP_THIS_AND_NEXT)
        set (_SKIP_THIS 0)
        string(COMPARE EQUAL ${_LIBRARY} "optimized" _SKIP_THIS)
        if (NOT _SKIP_THIS AND NOT _SKIP_THIS_AND_NEXT)
            set(PLUGINAPI_LIBS "${PLUGINAPI_LIBS} ${_LIBRARY} ")
        endif (NOT _SKIP_THIS AND NOT _SKIP_THIS_AND_NEXT)
    endif (_SKIP_THIS_AND_NEXT)
endforeach(_LIBRARY ${HOOMD_COMMON_LIBS})

# add cudart (if CUDA is enabled)
if (ENABLE_CUDA)
set(PLUGINAPI_LIBS "${PLUGINAPI_LIBS} ${CUDA_LIBRARIES} ")
endif (ENABLE_CUDA)

## build a list of include directories needed
set (PLUGINAPI_CFLAGS "")
set(PLUGINAPI_CFLAGS "${PLUGINAPI_CFLAGS} -I\${HOOMD_ROOT}/include/hoomd ")
set(PLUGINAPI_CFLAGS "${PLUGINAPI_CFLAGS} -isystem ${Boost_INCLUDE_DIR} ")
set(PLUGINAPI_CFLAGS "${PLUGINAPI_CFLAGS} -I${PYTHON_INCLUDE_PATH} ")

if (ENABLE_CUDA)
set(PLUGINAPI_CFLAGS "${PLUGINAPI_CFLAGS} -I${CUDA_INCLUDE_DIRS} ")
endif (ENABLE_CUDA)

# configure the compiler options script and install it
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd-config.sh.in ${CMAKE_CURRENT_BINARY_DIR}/hoomd-config.sh @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd-config.sh DESTINATION ${BIN_INSTALL_DIR})

endif (ENABLE_PLUGINAPI)
