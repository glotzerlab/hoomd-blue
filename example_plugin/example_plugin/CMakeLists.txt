# Maintainer: mspells

set(COMPONENT_NAME example_plugin)

set(_${COMPONENT_NAME}_sources
    module.cc
    ExampleUpdater.cc
    )

set(_${COMPONENT_NAME}_cu_sources
    ExampleUpdater.cu
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd)

fix_cudart_rpath(_${COMPONENT_NAME})

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${COMPONENT_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    update.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
