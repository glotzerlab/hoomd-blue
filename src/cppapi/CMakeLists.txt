# $Id$
# $URL$

#################################
## write a C++ include file and a config.h file
option(ENABLE_CXX "Create headers and library variables to link c++ codes to the hoomd library" OFF)
mark_as_advanced(ENABLE_CXX)
if (ENABLE_CXX)
# set up a config.h file to include the appropriate precompiler directives
configure_file(${HOOMD_SOURCE_DIR}/cppapi/Config.h.cmake ${HOOMD_BINARY_DIR}/cppapi/config.h)
## set up a "hoomd.h" header file that includes the "config.h" file as well as the other headers 
file(WRITE ${HOOMD_BINARY_DIR}/cppapi/hoomd.h "#ifndef _HOOMD_H\n#define _HOOMD_H\n\n")
file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd.h "#include \"config.h\"\n") 
file(GLOB HEADER_FILES "${HOOMD_SOURCE_DIR}/*/*.h")
foreach(_FILENAME ${HEADER_FILES})
    get_filename_component(_FILENAMENOPATH ${_FILENAME} NAME)
    string(REGEX MATCH "GPU" _MATCH "${_FILENAME}")
    set (_DONT_SKIP_THIS 1)
    if (NOT ENABLE_CUDA)
        set (_DONT_SKIP_THIS 0)
        string(COMPARE EQUAL "${_MATCH}" "" _DONT_SKIP_THIS)
    endif (NOT ENABLE_CUDA)
    if (_DONT_SKIP_THIS)
        file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd.h "#include \"${_FILENAMENOPATH}\"\n")
    endif (_DONT_SKIP_THIS)
endforeach(_FILENAME ${HEADER_FILES})

#if (ENABLE_CUDA)
#file(GLOB HEADER_FILES "${HOOMD_SOURCE_DIR}/*/*.cuh")
#foreach(_FILENAME ${HEADER_FILES})
#    get_filename_component(_FILENAMENOPATH ${_FILENAME} NAME)
#    file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd.h "#include \"${_FILENAMENOPATH}\"\n")
#endforeach(_FILENAME ${HEADER_FILES})
#endif (ENABLE_CUDA)

file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd.h "\n#endif\n\n")

# copy the headers into the include/hoomd folder
file(GLOB files "${HOOMD_SOURCE_DIR}/*/*h")
install(FILES ${files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hoomd)
file(GLOB files "${HOOMD_BINARY_DIR}/cppapi/*.h")
install(FILES ${files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hoomd)

## write a list of libraries used for linking.
file(WRITE ${HOOMD_BINARY_DIR}/cppapi/hoomd_linker_setup.sh "#!/bin/bash\n\n")
file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd_linker_setup.sh "HOOMD_LINK_LIBS=\" \\\n")
#message(STATUS ${HOOMD_COMMON_LIBS})
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
            file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd_linker_setup.sh "${_LIBRARY} \\\n")
        endif (NOT _SKIP_THIS AND NOT _SKIP_THIS_AND_NEXT)
    endif (_SKIP_THIS_AND_NEXT)
endforeach(_LIBRARY ${HOOMD_COMMON_LIBS})
file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd_linker_setup.sh "${CMAKE_INSTALL_PREFIX}/lib/libhoomd_python_module.a \\\n")
file(APPEND ${HOOMD_BINARY_DIR}/cppapi/hoomd_linker_setup.sh "-I${CMAKE_INSTALL_PREFIX}/include/hoomd\"")
endif (ENABLE_CXX)
