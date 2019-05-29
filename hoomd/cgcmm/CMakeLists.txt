set(PACKAGE_NAME cgcmm)

set(_${PACKAGE_NAME}_sources
     module-${PACKAGE_NAME}.cc
     CGCMMAngleForceCompute.cc
     CGCMMForceCompute.cc
   )

set(_${PACKAGE_NAME}_headers
    CGCMMAngleForceComputeGPU.h
    CGCMMAngleForceCompute.h
    CGCMMAngleForceGPU.cuh
    CGCMMForceComputeGPU.h
    CGCMMForceCompute.h
    CGCMMForceGPU.cuh
    )


if (ENABLE_CUDA)
list(APPEND _${PACKAGE_NAME}_sources
    CGCMMAngleForceComputeGPU.cc
    CGCMMForceComputeGPU.cc
    )
endif()

set(_${PACKAGE_NAME}_cu_sources
    CGCMMAngleForceGPU.cu
    CGCMMForceGPU.cu
   )

if (ENABLE_CUDA)
set(_cuda_sources ${_${PACKAGE_NAME}_cu_sources})
set_source_files_properties(${_cuda_sources} PROPERTIES COMPILE_DEFINITIONS "NVCC")
endif (ENABLE_CUDA)

pybind11_add_module(_${PACKAGE_NAME} SHARED ${_${PACKAGE_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
if (APPLE)
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_${PACKAGE_NAME} PUBLIC _md)

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
          angle.py
          pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PACKAGE_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${PACKAGE_NAME} ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_${PACKAGE_NAME}_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/${PACKAGE_NAME}
       )

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
