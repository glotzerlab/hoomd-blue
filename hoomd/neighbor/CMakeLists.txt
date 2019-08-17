# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

set(COMPONENT_NAME neighbor)

if (NOT ENABLE_CUDA)
    message(FATAL_ERROR "CUDA must be enabled to use ${COMPONENT_NAME} module")
endif(NOT ENABLE_CUDA)

##############################
# List of all source files
# c++ source files
set(_cc_sources
    LBVH.cc
    LBVHTraverser.cc
    )
# cuda source files
set(_cu_sources
    LBVH.cu
    LBVHTraverser.cu
    )
# headers
set(_h_sources
    BoundingVolumes.h
    InsertOps.h
    LBVH.cuh
    LBVH.h
    LBVHTraverser.cuh
    LBVHTraverser.h
    OutputOps.h
    QueryOps.h
    TransformOps.h
    )
###########################

# compile cuda sources
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)

# setup library with cc sources
add_library(${COMPONENT_NAME} SHARED ${_cc_sources} ${_CUDA_GENERATED_FILES})
target_link_libraries(${COMPONENT_NAME} PRIVATE ${HOOMD_LIBRARIES})
set_target_properties(${COMPONENT_NAME} PROPERTIES PUBLIC_HEADER "${_h_sources}")
if (APPLE)
set_target_properties(${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN")
endif()

# if we are compiling with MPI support built in, set appropriate compiler/linker flags
if (ENABLE_MPI)
   if(MPI_CXX_COMPILE_FLAGS)
       set_target_properties(${COMPONENT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_CXX_COMPILE_FLAGS)
   if(MPI_CXX_LINK_FLAGS)
       set_target_properties(${COMPONENT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_CXX_LINK_FLAGS)
endif(ENABLE_MPI)
fix_cudart_rpath(${COMPONENT_NAME})

# install the library
install(TARGETS ${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
        PUBLIC_HEADER DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/${COMPONENT_NAME})

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)
