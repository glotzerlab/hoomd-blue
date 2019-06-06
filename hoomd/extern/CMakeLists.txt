file(GLOB_RECURSE _extern_headers
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     *.h *.cuh *.hpp *.cmake)

foreach(file ${_extern_headers})
    # install into installation target
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/${dir})
endforeach()
