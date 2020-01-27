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

if (ENABLE_HIP)
    foreach(src ${_dem_cu_sources_raw})
      list(APPEND _dem_cu_sources templateInstances/${src})
    endforeach()
endif(ENABLE_HIP)

if (ENABLE_HIP)
set(_cuda_sources ${_dem_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_dem SHARED ${_dem_sources} ${_cuda_sources} ${_dem_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_dem ALIAS _dem)

if (APPLE)
set_target_properties(_dem PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_dem PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()
# link the library to its dependencies
target_link_libraries(_dem PUBLIC _md)

fix_cudart_rpath(_dem)

# install the library
install(TARGETS _dem
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/dem
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
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/dem
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_dem ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_dem_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/dem
       )

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
