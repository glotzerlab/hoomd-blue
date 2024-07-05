set(PACKAGE_NAME metal)

set(_${PACKAGE_NAME}_sources
     module-${PACKAGE_NAME}.cc
     EAMForceCompute.cc
   )

set(_${PACKAGE_NAME}_headers EAMForceComputeGPU.h
                             EAMForceCompute.h
   )

if (ENABLE_HIP)
list(APPEND _${PACKAGE_NAME}_sources
     EAMForceComputeGPU.cc
    )
endif()

set(_${PACKAGE_NAME}_cu_sources
     EAMForceGPU.cu
   )

if (ENABLE_HIP)
set(_cuda_sources ${_${PACKAGE_NAME}_cu_sources})
endif (ENABLE_HIP)

hoomd_add_module(_${PACKAGE_NAME} SHARED ${_${PACKAGE_NAME}_sources} ${_cuda_sources} ${_${PACKAGE_NAME}_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_${PACKAGE_NAME} ALIAS _${PACKAGE_NAME})

if (APPLE)
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_${PACKAGE_NAME} PUBLIC _md)

# install the library
install(TARGETS _${PACKAGE_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${PACKAGE_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files __init__.py
          pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${PACKAGE_NAME}
       )

copy_files_to_build("${files}" "${PACKAGE_NAME}" "*.py")

# install headers in installation target
install(FILES ${_${PACKAGE_NAME}_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/${PACKAGE_NAME}
       )

if (BUILD_TESTING)
    # add_subdirectory(test-py)
    # add_subdirectory(test)
endif()
