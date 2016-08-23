
file(GLOB_RECURSE _extern_headers
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     *.h *.cuh *.hpp)

# cereal should get its headers installed a separate way, so find and
# remove those from _extern_headers
file(GLOB_RECURSE _cereal_headers
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/cereal
     cereal/include/*.h cereal/include/*.cuh cereal/include/*.hpp)

list(REMOVE_ITEM _extern_headers ${_cereal_headers})

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

# copy python modules to the build directory to make it a working python package
MACRO(copy_cereal_header file)
    add_custom_command (
        OUTPUT include/${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cereal/${file} ${CMAKE_CURRENT_BINARY_DIR}/../${file}
        COMMENT    "Copy hoomd/${file}"
    )
ENDMACRO(copy_cereal_header)


foreach(file ${_cereal_headers})
    # copy headers into build directory
    copy_cereal_header(${file})
    # install into installation target
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cereal/${file} DESTINATION ${PYTHON_MODULE_BASE_DIR}/${dir})
    list(APPEND _cereal_installed_headers include/${file})
endforeach()

add_custom_target(copy_extern ALL DEPENDS ${_extern_headers} ${_cereal_installed_headers})
