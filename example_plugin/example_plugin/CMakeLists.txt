# Maintainer: mspells

set(_${PROJECT_NAME}_sources
    module.cc
    ExampleUpdater.cc
    )

set(_${PROJECT_NAME}_cu_sources
    ExampleUpdater.cu
    )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_${PROJECT_NAME}_sources} ${_${PROJECT_NAME}_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${PROJECT_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_${PROJECT_NAME} SHARED ${_${PROJECT_NAME}_sources} ${_CUDA_GENERATED_FILES})

# link the library to its dependencies
target_link_libraries(_${PROJECT_NAME} ${HOOMD_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_${PROJECT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_${PROJECT_NAME}")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_${PROJECT_NAME} PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_${PROJECT_NAME})

# install the library
install(TARGETS _${PROJECT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PROJECT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${PROJECT_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    update.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PROJECT_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${PROJECT_NAME} ALL DEPENDS ${files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
