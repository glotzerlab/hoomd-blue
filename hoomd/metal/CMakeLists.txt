set(PACKAGE_NAME metal)

set(_${PACKAGE_NAME}_sources
     module-${PACKAGE_NAME}.cc
     EAMForceCompute.cc
   )

set(_${PACKAGE_NAME}_headers EAMForceComputeGPU.h
                             EAMForceCompute.h
   )

if (ENABLE_CUDA)
list(APPEND _${PACKAGE_NAME}_sources
     EAMForceComputeGPU.cc
    )
endif()

set(_${PACKAGE_NAME}_cu_sources
     EAMForceGPU.cu
   )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_${PACKAGE_NAME}_sources} ${_${PACKAGE_NAME}_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${PACKAGE_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_${PACKAGE_NAME} SHARED ${_${PACKAGE_NAME}_sources} ${_CUDA_GENERATED_FILES})

# link the library to its dependencies
target_link_libraries(_${PACKAGE_NAME} _hoomd _md ${HOOMD_COMMON_LIBS})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_${PACKAGE_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_${PACKAGE_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_${PACKAGE_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_${PACKAGE_NAME}")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_${PACKAGE_NAME} PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_${PACKAGE_NAME})

# install the library
install(TARGETS _${PACKAGE_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PACKAGE_NAME}
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
        COMMENT    "Copy hoomd/${PACKAGE_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PACKAGE_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${PACKAGE_NAME} ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/${PACKAGE_NAME}/${file}
        COMMENT    "Copy hoomd/include/hoomd/${PACKAGE_NAME}/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_${PACKAGE_NAME}_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/${PACKAGE_NAME}
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_${PACKAGE_NAME}_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_${PACKAGE_NAME}_headers ALL DEPENDS ${_${PACKAGE_NAME}_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    # add_subdirectory(test)
endif()
