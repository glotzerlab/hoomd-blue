
if(NOT BUILD_MD)
    message(FATAL_ERROR "MPCD package cannot be built without MD.")
endif()

set(_mpcd_cc_sources
    module.cc
    ATCollisionMethod.cc
    CellCommunicator.cc
    CellThermoCompute.cc
    CellList.cc
    CollisionMethod.cc
    Communicator.cc
    Integrator.cc
    ManualVirtualParticleFiller.cc
    ParallelPlateGeometryFiller.cc
    PlanarPoreGeometryFiller.cc
    Sorter.cc
    SRDCollisionMethod.cc
    StreamingGeometry.cc
    StreamingMethod.cc
    VirtualParticleFiller.cc
    )

set(_mpcd_headers
    ATCollisionMethod.h
    BounceBackNVE.h
    BulkGeometry.h
    BulkStreamingMethod.h
    CellCommunicator.h
    CellThermoCompute.h
    CellList.h
    CollisionMethod.h
    BounceBackStreamingMethod.h
    Communicator.h
    CommunicatorUtilities.h
    Integrator.h
    ManualVirtualParticleFiller.h
    ParticleData.h
    ParticleDataSnapshot.h
    ParticleDataUtilities.h
    ParallelPlateGeometry.h
    ParallelPlateGeometryFiller.h
    PlanarPoreGeometry.h
    PlanarPoreGeometryFiller.h
    Sorter.h
    SRDCollisionMethod.h
    StreamingGeometry.h
    StreamingMethod.h
    VirtualParticleFiller.h
    )

set(_mpcd_cu_sources "")

if(ENABLE_HIP)
    list(APPEND _mpcd_cc_sources
        ATCollisionMethodGPU.cc
        CellThermoComputeGPU.cc
        CellListGPU.cc
        CommunicatorGPU.cc
        ParallelPlateGeometryFillerGPU.cc
        PlanarPoreGeometryFillerGPU.cc
        SorterGPU.cc
        SRDCollisionMethodGPU.cc
        )
    list(APPEND _mpcd_headers
        ATCollisionMethodGPU.cuh
        ATCollisionMethodGPU.h
        BounceBackNVEGPU.cuh
        BounceBackNVEGPU.h
        BulkStreamingMethodGPU.h
        CellCommunicator.cuh
        CellThermoComputeGPU.cuh
        CellThermoComputeGPU.h
        CellListGPU.cuh
        CellListGPU.h
        CommunicatorGPU.cuh
        CommunicatorGPU.h
        BounceBackStreamingMethodGPU.cuh
        BounceBackStreamingMethodGPU.h
        ParticleData.cuh
        ParallelPlateGeometryFillerGPU.cuh
        ParallelPlateGeometryFillerGPU.h
        PlanarPoreGeometryFillerGPU.cuh
        PlanarPoreGeometryFillerGPU.h
        SorterGPU.cuh
        SorterGPU.h
        SRDCollisionMethodGPU.cuh
        SRDCollisionMethodGPU.h
        )
    set(_mpcd_cu_sources
        ATCollisionMethodGPU.cu
        BounceBackNVEGPU.cu
        CellThermoComputeGPU.cu
        CellListGPU.cu
        CommunicatorGPU.cu
        ParticleData.cu
        ParallelPlateGeometryFillerGPU.cu
        PlanarPoreGeometryFillerGPU.cu
        SorterGPU.cu
        SRDCollisionMethodGPU.cu
        )
endif()

# generate cc and cu templates for forces and streaming geometries
set(_forces
    BlockForce
    ConstantForce
    SineForce
    NoForce
    )
set(_geometries
    ParallelPlateGeometry
    PlanarPoreGeometry
    )
foreach(_force ${_forces})
    # append force itself first
    list(APPEND _mpcd_cc_sources ${_force}.cc)
    list(APPEND _mpcd_headers ${_force}.h)

    # bulk geometry is special
    configure_file(
        BulkStreamingMethod.cc.inc
        BulkStreamingMethod${_force}.cc
        @ONLY
        )
    list(APPEND _mpcd_cc_sources BulkStreamingMethod${_force}.cc)
    if(ENABLE_HIP)
        configure_file(
            BulkStreamingMethodGPU.cc.inc
            BulkStreamingMethod${_force}GPU.cc
            @ONLY
        )
        configure_file(
            BulkStreamingMethodGPU.cu.inc
            BulkStreamingMethod${_force}GPU.cu
            @ONLY
        )
        list(APPEND _mpcd_cc_sources BulkStreamingMethod${_force}GPU.cc)
        list(APPEND _mpcd_cu_sources BulkStreamingMethod${_force}GPU.cu)
    endif()

    # then the bounce back geometries
    foreach(_geometry ${_geometries})
        configure_file(
            BounceBackStreamingMethod.cc.inc
            BounceBackStreamingMethod${_geometry}${_force}.cc
            @ONLY
            )
        list(APPEND _mpcd_cc_sources BounceBackStreamingMethod${_geometry}${_force}.cc)
        if(ENABLE_HIP)
            configure_file(
                BounceBackStreamingMethodGPU.cc.inc
                BounceBackStreamingMethod${_geometry}${_force}GPU.cc
                @ONLY
            )
            configure_file(
                BounceBackStreamingMethodGPU.cu.inc
                BounceBackStreamingMethod${_geometry}${_force}GPU.cu
                @ONLY
            )
            list(APPEND _mpcd_cc_sources BounceBackStreamingMethod${_geometry}${_force}GPU.cc)
            list(APPEND _mpcd_cu_sources BounceBackStreamingMethod${_geometry}${_force}GPU.cu)
        endif()
    endforeach()
endforeach()

if(ENABLE_HIP)
    set_source_files_properties(${_mpcd_cu_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif (ENABLE_HIP)

hoomd_add_module(_mpcd SHARED ${_mpcd_cc_sources} ${_mpcd_cu_sources} ${_mpcd_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_mpcd ALIAS _mpcd)

if(APPLE)
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_mpcd PUBLIC _hoomd _md)

# install the library
install(TARGETS _mpcd EXPORT HOOMDTargets LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd)
install(FILES ${_mpcd_headers} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/mpcd)

# install and also copy python modules to the build directory to make it a working python package
set(files
    __init__.py
    collide.py
    fill.py
    force.py
    geometry.py
    integrate.py
    methods.py
    stream.py
    tune.py
    )
install(FILES ${files} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd)
copy_files_to_build("${files}" "mpcd" "*.py")

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
add_subdirectory(pytest)
