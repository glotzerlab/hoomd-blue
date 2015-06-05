file(GLOB _SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB _CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

list(APPEND PLUGIN_SRC ${_SRCS})

if (ENABLE_CUDA)
    CUDA_COMPILE(_CUDA_GENERATED_FILES ${_CUDA_SRCS} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
    list(APPEND PLUGIN_SRC ${_CUDA_GENERATED_FILES})
endif (ENABLE_CUDA)

# build the C++ python module
set(LIB_TARGET "_${PROJECT_NAME}")
add_library(${LIB_TARGET} ${PLUGIN_SRC} ${PYMODULE_BUILD})
target_link_libraries(${LIB_TARGET} ${HOOMD_LIBRARIES})

if (ENABLE_MPI)
    # if we are compiling with MPI support built in, set appropriate
    # compiler/linker flags
    if(MPI_COMPILE_FLAGS)
        set_target_properties(${LIB_TARGET} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
    endif(MPI_COMPILE_FLAGS)
    if(MPI_LINK_FLAGS)
        set_target_properties(${LIB_TARGET} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
    endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

# fix up the file name so it will work as a python module
set_target_properties(${LIB_TARGET} PROPERTIES PREFIX "" OUTPUT_NAME "${LIB_TARGET}")
if(APPLE)
    set_target_properties(${LIB_TARGET} PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(${LIB_TARGET})

# install the library to the hoomd_plugins directory
install(TARGETS ${LIB_TARGET}
        LIBRARY DESTINATION hoomd_plugins/${PROJECT_NAME}
        )
