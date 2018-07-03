# Maintainer: mspells

set(_dem_sources module.cc)

set(_dem_cpu_sources DEM2DForce_NF_SWCA.cc
                     DEM2DForce_NF_WCA.cc
                     DEM3DForce_NF_SWCA.cc
                     DEM3DForce_NF_WCA.cc)

set(_dem_cu_sources_raw DEM2DForce_NF_SWCA.cu
                        DEM2DForce_NF_WCA.cu
                        DEM3DForce_NF_SWCA.cu
                        DEM3DForce_NF_WCA.cu)

set(_dem_headers
    atomics.cuh
    DEM2DForceComputeGPU.h
    DEM2DForceCompute.h
    DEM2DForceGPU.cuh
    DEM3DForceComputeGPU.h
    DEM3DForceCompute.h
    DEM3DForceGPU.cuh
    DEMEvaluator.h
    NoFriction.h
    SWCAPotential.h
    VectorMath.h
    WCAPotential.h
    )

foreach(src ${_dem_cpu_sources})
  list(APPEND _dem_sources templateInstances/${src})
endforeach()

if (ENABLE_CUDA)
    foreach(src ${_dem_cu_sources_raw})
      list(APPEND _dem_cu_sources templateInstances/${src})
    endforeach()
endif(ENABLE_CUDA)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_dem_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module(_dem SHARED ${_dem_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)
if (APPLE)
set_target_properties(_dem PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_dem PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()
# link the library to its dependencies
target_link_libraries(_dem PRIVATE _md ${HOOMD_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_dem PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_dem PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

fix_cudart_rpath(_dem)

# install the library
install(TARGETS _dem
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/dem
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
        COMMENT    "Copy hoomd/dem/${file}"
    )
ENDMACRO(copy_file)

set(files   __init__.py
            pair.py
            params.py
            utils.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/dem
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_dem ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/dem/${file}
        COMMENT    "Copy hoomd/include/hoomd/dem/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_dem_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/dem
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_dem_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_dem_headers ALL DEPENDS ${_dem_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
