
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
    ExternalField.cc
    Integrator.cc
    SlitGeometryFiller.cc
    SlitPoreGeometryFiller.cc
    Sorter.cc
    SRDCollisionMethod.cc
    StreamingGeometry.cc
    StreamingMethod.cc
    VirtualParticleFiller.cc
    )

set(_mpcd_headers
    ATCollisionMethod.h
    BounceBackNVE.h
    BoundaryCondition.h
    BulkGeometry.h
    CellCommunicator.h
    CellThermoCompute.h
    CellList.h
    CollisionMethod.h
    ConfinedStreamingMethod.h
    Communicator.h
    CommunicatorUtilities.h
    ExternalField.h
    Integrator.h
    ParticleData.h
    ParticleDataSnapshot.h
    ParticleDataUtilities.h
    SlitGeometry.h
    SlitGeometryFiller.h
    SlitPoreGeometry.h
    SlitPoreGeometryFiller.h
    Sorter.h
    SRDCollisionMethod.h
    StreamingGeometry.h
    StreamingMethod.h
    VirtualParticleFiller.h
    )

if (ENABLE_HIP)
list(APPEND _mpcd_sources
    ATCollisionMethodGPU.cc
    CellThermoComputeGPU.cc
    CellListGPU.cc
    CommunicatorGPU.cc
    SlitGeometryFillerGPU.cc
    SlitPoreGeometryFillerGPU.cc
    SorterGPU.cc
    SRDCollisionMethodGPU.cc
    )
list(APPEND _mpcd_headers
    ATCollisionMethodGPU.cuh
    ATCollisionMethodGPU.h
    BounceBackNVEGPU.cuh
    BounceBackNVEGPU.h
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
    SlitPoreGeometryFillerGPU.cuh
    SlitPoreGeometryFillerGPU.h
    SorterGPU.cuh
    SorterGPU.h
    SRDCollisionMethodGPU.cuh
    SRDCollisionMethodGPU.h
    )
endif()

set(_mpcd_cu_sources
    ATCollisionMethodGPU.cu
    BounceBackNVEGPU.cu
    CellThermoComputeGPU.cu
    CellListGPU.cu
    ConfinedStreamingMethodGPU.cu
    CommunicatorGPU.cu
    ExternalField.cu
    ParticleData.cu
    SlitGeometryFillerGPU.cu
    SlitPoreGeometryFillerGPU.cu
    SorterGPU.cu
    SRDCollisionMethodGPU.cu
    )

if (ENABLE_HIP)
    set(_cuda_sources ${_mpcd_cu_sources})
endif (ENABLE_HIP)

hoomd_add_module(_mpcd SHARED ${_mpcd_sources} ${LINK_OBJ} ${_cuda_sources} ${_mpcd_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_mpcd ALIAS _mpcd)

if (APPLE)
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()

if (ENABLE_HIP)
    # Separable compilation is needed to support ExternalField.cu polymorphism.
    # This likely breaks plugin support though. :-(
    set_property(TARGET _mpcd PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# link the library to its dependencies
target_link_libraries(_mpcd PUBLIC _md)
if (ENABLE_HIP)
    target_link_libraries(_mpcd PUBLIC CUDA::cudadevrt)
endif (ENABLE_HIP)

# install the library
install(TARGETS _mpcd EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files
    __init__.py
    collide.py
    force.py
    integrate.py
    stream.py
    update.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd
       )

copy_files_to_build("${files}" "mpcd" "*.py")

# install headers in installation target
install(FILES ${_mpcd_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/mpcd
       )

if (BUILD_TESTING)
     add_subdirectory(test)
endif()
add_subdirectory(pytest)
if (BUILD_VALIDATION)
    # add_subdirectory(validation)
endif (BUILD_VALIDATION)
