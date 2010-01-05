# $Id$
# $URL$
# Maintainer: joaander

#################################
## write a C++ include file and a hoomd_config.h file
option(ENABLE_PLUGINAPI "Create headers and library variables to link c++ codes to the hoomd library" OFF)

if (ENABLE_PLUGINAPI)
# set up a hoomd_config.h file to include the appropriate precompiler directives
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/hoomd_config.h)
## set up a "hoomd.h" header file that includes the "hoomd_config.h" file as well as the other headers 
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hoomd.h "#ifndef _HOOMD_H\n#define _HOOMD_H\n\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hoomd.h "#include \"hoomd_config.h\"\n") 
file(GLOB HEADER_FILES "${HOOMD_SOURCE_DIR}/analyzers/*.h" "${HOOMD_SOURCE_DIR}/computes*/*.h" 
                       "${HOOMD_SOURCE_DIR}/data_structures/*.h" "${HOOMD_SOURCE_DIR}/extern/*.h" 
                       "${HOOMD_SOURCE_DIR}/system/*.h" "${HOOMD_SOURCE_DIR}/updaters*/*.h"
                       "${HOOMD_SOURCE_DIR}/utils/*.h" "${HOOMD_BINARY_DIR}/utils/HOOMDVersion.h"
                       "${HOOMD_SOURCE_DIR}/potentials/*.h")
foreach(_FILENAME ${HEADER_FILES})
    get_filename_component(_FILENAMENOPATH ${_FILENAME} NAME)
    string(REGEX MATCH "GPU" _MATCH "${_FILENAME}")
    set (_DONT_SKIP_THIS 1)
    if (NOT ENABLE_CUDA)
        set (_DONT_SKIP_THIS 0)
        string(COMPARE EQUAL "${_MATCH}" "" _DONT_SKIP_THIS)
    endif (NOT ENABLE_CUDA)
    if (_DONT_SKIP_THIS)
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hoomd.h "#include \"${_FILENAMENOPATH}\"\n")
    endif (_DONT_SKIP_THIS)
endforeach(_FILENAME ${HEADER_FILES})

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hoomd.h "\n#endif\n\n")

# copy the headers into the include/hoomd folder
file(GLOB files "${HOOMD_SOURCE_DIR}/*/*h" "${HOOMD_BINARY_DIR}/utils/HOOMDVersion.h")
install(FILES ${files} DESTINATION ${INC_INSTALL_DIR})
file(GLOB files "${CMAKE_CURRENT_BINARY_DIR}/*.h")
install(FILES ${files} DESTINATION ${INC_INSTALL_DIR})

## build a list of libraries used for linking.
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
