# Maintainer: joaander

set(_deprecated_sources
     module-deprecated.cc
     MSDAnalyzer.cc
     HOOMDDumpWriter.cc
     POSDumpWriter.cc
     HOOMDInitializer.cc
     RandomGenerator.cc
     xmlParser.cc
   )

set(_deprecated_headers
    HOOMDDumpWriter.h
    HOOMDInitializer.h
    MSDAnalyzer.h
    POSDumpWriter.h
    RandomGenerator.h
    xmlParser.h
    )

if (ENABLE_CUDA)
list(APPEND _deprecated_sources
    )
endif()

set(_deprecated_cu_sources
   )

if (ENABLE_CUDA)
set(_cuda_sources ${_deprecated_cu_sources})
set_source_files_properties(${_cuda_sources} PROPERTIES COMPILE_DEFINITIONS "NVCC")
endif (ENABLE_CUDA)

pybind11_add_module(_deprecated SHARED ${_deprecated_sources} ${_cuda_sources} NO_EXTRAS)
if (APPLE)
set_target_properties(_deprecated PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_deprecated PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_deprecated PUBLIC _hoomd)

fix_cudart_rpath(_deprecated)

# install the library
install(TARGETS _deprecated
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/deprecated
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
        COMMENT    "Copy hoomd/deprecated/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          analyze.py
          dump.py
          init.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/deprecated
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_deprecated ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_deprecated_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/deprecated
       )

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
