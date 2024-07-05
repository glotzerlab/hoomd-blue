# Template CMakeLists.txt for plugins/components.

set(COMPONENT_NAME updater_plugin)

# Specify any C++ sources
set(_${COMPONENT_NAME}_sources
    module.cc
    ExampleUpdater.cc
    )

# Specify any CUDA sources
set(_${COMPONENT_NAME}_cu_sources
    ExampleUpdater.cu
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
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
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd)

# Install the library.
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# Copy python modules to the build directory to make it a working python package. Any files that
# should be copied to the install directory should be listed here.set(files
set(files
    __init__.py
    update.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

copy_files_to_build("${files}" "${COMPONENT_NAME}" "*.py")

# Python tests.
add_subdirectory(pytest)
