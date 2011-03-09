## $Id$
## $URL$
# Maintainer: joaander

#################
macro(add_libhoomd_source_dir dir)
add_subdirectory(${dir})

# find all source files in this directory
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cc)
file(GLOB CU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cu)
file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cuh)

# add them to the source group
source_group(${dir} FILES ${CU_SRCS} ${SRCS} ${HDRS})

# add them to the main libhoomd list
list(APPEND _libhoomd_sources ${SRCS} ${HDRS})
list(APPEND _libhoomd_cu_sources ${CU_SRCS})
endmacro(add_libhoomd_source_dir)
#################

set(_libhoomd_sources "")
set(_libhoomd_cu_sources "")

# process all subdirectories, which add source files to _libhoomd_sources and _libhoomd_cu_sources
set(_dir_list analyzers computes data_structures extern potentials python system updaters utils)
if (ENABLE_CUDA)
list(APPEND _dir_list computes_gpu cuda updaters_gpu)
endif (ENABLE_CUDA)

foreach (dir ${_dir_list})
add_libhoomd_source_dir(${dir})
endforeach (dir)

if (ENABLE_CUDA)
if (BUILD_SHARED_LIBS)
set (_libhoomd_shared SHARED)
endif (BUILD_SHARED_LIBS)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_libhoomd_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} ${_libhoomd_shared})
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
        set_target_properties(libhoomd PROPERTIES SUFFIX ".so")
    endif(APPLE)
    fix_cudart_rpath(libhoomd)

    # if this is a PYTHON_SITEDIR build, set the rpath appropriately
    if (PYTHON_SITEDIR)
        set_target_properties(libhoomd PROPERTIES INSTALL_RPATH "\\\$ORIGIN/../../hoomd")
    endif (PYTHON_SITEDIR)
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
        LIBRARY DESTINATION ${HOOMD_PYTHON_MODULE_DIR}
        ARCHIVE DESTINATION ${LIB_BASE_INSTALL_DIR}
        )

add_subdirectory(api)

