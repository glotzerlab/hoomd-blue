# Maintainer: joaander

set(_deprecated_sources
     module-deprecated.cc
     MSDAnalyzer.cc
     HOOMDDumpWriter.cc
     POSDumpWriter.cc
     HOOMDInitializer.cc
     RandomGenerator.cc
   )

set(_deprecated_headers
    HOOMDDumpWriter.h
    HOOMDInitializer.h
    MSDAnalyzer.h
    POSDumpWriter.h
    RandomGenerator.h
    )

if (ENABLE_CUDA)
list(APPEND _deprecated_sources
    )
endif()

set(_deprecated_cu_sources
   )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_deprecated_sources} ${_deprecated_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_deprecated_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_deprecated SHARED ${_deprecated_sources} ${_CUDA_GENERATED_FILES})

# link the library to its dependencies
target_link_libraries(_deprecated _hoomd ${HOOMD_COMMON_LIBS})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_deprecated PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_deprecated PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_deprecated PROPERTIES PREFIX "" OUTPUT_NAME "_deprecated")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_deprecated PROPERTIES SUFFIX ".so")
endif(APPLE)
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

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/deprecated/${file}
        COMMENT    "Copy hoomd/include/hoomd/deprecated/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_deprecated_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/deprecated
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_deprecated_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_deprecated_headers ALL DEPENDS ${_deprecated_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
