# Template CMakeLists.txt for plugins/components.

set(COMPONENT_NAME shape_plugin)

# Check that the HOOMD build configuration supports plugin dependencies
if(NOT BUILD_HPMC)
    message(WARNING "The HOOMD build configuration does not support HPMC. Skipping build of "
            "plugin component ${COMPONENT_NAME}.")
    return()
endif()

# Specify any C++ sources
set(_${COMPONENT_NAME}_sources
    module.cc
    )

# Specify any CUDA sources
set(_${COMPONENT_NAME}_cu_sources
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources}
    kernels.cu
    )
endif (ENABLE_HIP)

hoomd_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
# Alias into the HOOMD namespace so that plugins and symlinked components both work.
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# Link the library to its dependencies. Add or remove HOOMD extension modules (and/or external C++
# libraries) as needed.
target_link_libraries(_${COMPONENT_NAME}
                      PUBLIC HOOMD::_hoomd
                      PUBLIC HOOMD::_hpmc
                      )

# Install the library.
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# Copy python modules to the build directory to make it a working python package. Any files that
# should be copied to the install directory should be listed here.
set(files
    __init__.py
    integrate.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

copy_files_to_build("${files}" "${COMPONENT_NAME}" "*.py")

# Python tests.
add_subdirectory(pytest)
