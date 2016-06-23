
file(GLOB_RECURSE _extern_headers
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     *.h *.cuh)

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/extern/${file}
        COMMENT    "Copy hoomd/include/hoomd/extern/${file}"
    )
ENDMACRO(copy_header)

foreach(file ${_extern_headers})
    # copy headers into build directory
    copy_header(${file})
    # install into installation target
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/extern/${dir})
endforeach()

add_custom_target(copy_extern ALL DEPENDS ${_extern_headers})
