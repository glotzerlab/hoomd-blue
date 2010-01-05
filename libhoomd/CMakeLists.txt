## $Id$
## $URL$
# Maintainer: joaander

set(_libhoomd_sources "")
set(_libhoomd_cu_sources "")

# process all subdirectories, which add source files to _libhoomd_sources
add_subdirectory(analyzers)
add_subdirectory(computes)
add_subdirectory(computes_gpu)
add_subdirectory(cuda)
add_subdirectory(data_structures)
add_subdirectory(extern)
add_subdirectory(potentials)
add_subdirectory(python)
add_subdirectory(system)
add_subdirectory(updaters)
add_subdirectory(updaters_gpu)
add_subdirectory(utils)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_libhoomd_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS})
source_group(cuda_generated FILES ${_CUDA_GENERATED_FILES})
endif (ENABLE_CUDA)

add_library (libhoomd ${_libhoomd_sources} ${_CUDA_GENERATED_FILES})

##################################
# change the name of the library if it is dynamic
if (NOT ENABLE_STATIC)
    # link the library to its dependancies
    target_link_libraries(libhoomd ${HOOMD_COMMON_LIBS})

    # tweak the properties of the output to make a functional python module
    set_target_properties(libhoomd PROPERTIES PREFIX "" OUTPUT_NAME "hoomd")

    # .dylib is not regognized as a python module by python on Mac OS X
    if(APPLE)
        set_target_properties(hoomd_python_module PROPERTIES SUFFIX ".so")
    endif(APPLE)
    fix_cudart_rpath(libhoomd)
endif (NOT ENABLE_STATIC)

if (ENABLE_STATIC)
    # change the library name to something more sensible
    if (WIN32)
        set_target_properties(libhoomd PROPERTIES OUTPUT_NAME "libhoomd")
    else(WIN32)
        set_target_properties(libhoomd PROPERTIES OUTPUT_NAME "hoomd")
    endif(WIN32)
endif (ENABLE_STATIC)

# install the library
install(TARGETS libhoomd
        LIBRARY DESTINATION ${LIB_INSTALL_DIR}/python-module
        ARCHIVE DESTINATION ${LIB_BASE_INSTALL_DIR}
        )

