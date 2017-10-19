# Maintainer: mphoward
if (NOT BUILD_MD)
    message(FATAL_ERROR "MPCD package cannot be built without MD.")
endif(NOT BUILD_MD)

set(_mpcd_sources
    module.cc
    ATCollisionMethod.cc
    CellCommunicator.cc
    CellThermoCompute.cc
    CellList.cc
    CollisionMethod.cc
    Communicator.cc
    Integrator.cc
    ParticleData.cc
    ParticleDataSnapshot.cc
    SlitGeometryFiller.cc
    Sorter.cc
    SRDCollisionMethod.cc
    StreamingGeometry.cc
    StreamingMethod.cc
    SystemData.cc
    SystemDataSnapshot.cc
    VirtualParticleFiller.cc
    )

set(_mpcd_headers
    ATCollisionMethod.h
    BoundaryCondition.h
    BulkGeometry.h
    CellCommunicator.h
    CellThermoCompute.h
    CellList.h
    CollisionMethod.h
    ConfinedStreamingMethod.h
    Communicator.h
    CommunicatorUtilities.h
    Integrator.h
    ParticleData.h
    ParticleDataSnapshot.h
    ParticleDataUtilities.h
    SlitGeometry.h
    SlitGeometryFiller.h
    Sorter.h
    SRDCollisionMethod.h
    StreamingGeometry.h
    StreamingMethod.h
    SystemData.h
    SystemDataSnapshot.h
    VirtualParticleFiller.h
    )

if (ENABLE_CUDA)
list(APPEND _mpcd_sources
    ATCollisionMethodGPU.cc
    CellThermoComputeGPU.cc
    CellListGPU.cc
    CommunicatorGPU.cc
    SlitGeometryFillerGPU.cc
    SorterGPU.cc
    SRDCollisionMethodGPU.cc
    )
list(APPEND _mpcd_headers
    ATCollisionMethodGPU.cuh
    ATCollisionMethodGPU.h
    CellCommunicator.cuh
    CellThermoComputeGPU.cuh
    CellThermoComputeGPU.h
    CellListGPU.cuh
    CellListGPU.h
    CommunicatorGPU.cuh
    CommunicatorGPU.h
    ConfinedStreamingMethodGPU.cuh
    ConfinedStreamingMethodGPU.h
    ParticleData.cuh
    SlitGeometryFillerGPU.cuh
    SlitGeometryFillerGPU.h
    SorterGPU.cuh
    SorterGPU.h
    SRDCollisionMethodGPU.cuh
    SRDCollisionMethodGPU.h
    )
endif()

set(_mpcd_cu_sources
    ATCollisionMethodGPU.cu
    CellThermoComputeGPU.cu
    CellListGPU.cu
    ConfinedStreamingMethodGPU.cu
    CommunicatorGPU.cu
    ParticleData.cu
    SlitGeometryFillerGPU.cu
    SorterGPU.cu
    SRDCollisionMethodGPU.cu
    )

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_mpcd_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module(_mpcd SHARED ${_mpcd_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)
if (APPLE)
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN/../md;$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_mpcd PRIVATE _md ${HOOMD_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_mpcd PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_mpcd PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

fix_cudart_rpath(_mpcd)

# install the library
install(TARGETS _mpcd
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/mpcd
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
        COMMENT    "Copy hoomd/mpcd/${file}"
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    collide.py
    data.py
    init.py
    stream.py
    update.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/mpcd
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_mpcd ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/mpcd/${file}
        COMMENT    "Copy hoomd/include/hoomd/mpcd/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_mpcd_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/mpcd
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_mpcd_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_mpcd_headers ALL DEPENDS ${_mpcd_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
add_subdirectory(test)
