if(NOT BUILD_MD)
    message(FATAL_ERROR "MPCD package cannot be built without MD.")
endif()

set(_mpcd_sources
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
    ReverseNonequilibriumShearFlow.cc
    Sorter.cc
    SRDCollisionMethod.cc
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
    ParallelPlateGeometryFiller.h
    PlanarPoreGeometryFiller.h
    RejectionVirtualParticleFiller.h
    ReverseNonequilibriumShearFlow.h
    ReverseNonequilibriumShearFlowUtilities.h
    Sorter.h
    SRDCollisionMethod.h
    StreamingMethod.h
    VirtualParticleFiller.h
    )

set(_forces
    BlockForce
    ConstantForce
    SineForce
    NoForce
    )

set(_geometries
    ConcentricCylindersGeometry
    CosineChannelGeometry
    CosineExpansionContractionGeometry
    ParallelPlateGeometry
    PlanarPoreGeometry
    SphereGeometry
    )

if(ENABLE_GPU)
    list(APPEND _mpcd_sources
        ATCollisionMethodGPU.cc
        ATCollisionMethodGPU.cu
        BounceBackNVEGPU.cu
        CellThermoComputeGPU.cc
        CellThermoComputeGPU.cu
        CellListGPU.cc
        CellListGPU.cu
        CollisionMethod.cu
        CommunicatorGPU.cc
        CommunicatorGPU.cu
        ParallelPlateGeometryFillerGPU.cc
        ParallelPlateGeometryFillerGPU.cu
        ParticleData.cu
        PlanarPoreGeometryFillerGPU.cc
        PlanarPoreGeometryFillerGPU.cu
        RejectionVirtualParticleFillerGPU.cu
        ReverseNonequilibriumShearFlowGPU.cc
        ReverseNonequilibriumShearFlowGPU.cu
        SorterGPU.cc
        SorterGPU.cu
        SRDCollisionMethodGPU.cc
        SRDCollisionMethodGPU.cu
        )
    list(APPEND _mpcd_headers
        ATCollisionMethodGPU.cuh
        ATCollisionMethodGPU.h
        BounceBackNVEGPU.cuh
        BounceBackNVEGPU.h
        BounceBackStreamingMethodGPU.cuh
        BounceBackStreamingMethodGPU.h
        BulkStreamingMethodGPU.h
        CellCommunicator.cuh
        CellThermoComputeGPU.cuh
        CellThermoComputeGPU.h
        CellListGPU.cuh
        CellListGPU.h
        CollisionMethod.cuh
        CommunicatorGPU.cuh
        CommunicatorGPU.h
        ParallelPlateGeometryFillerGPU.cuh
        ParallelPlateGeometryFillerGPU.h
        ParticleData.cuh
        PlanarPoreGeometryFillerGPU.cuh
        PlanarPoreGeometryFillerGPU.h
        RejectionVirtualParticleFillerGPU.cuh
        RejectionVirtualParticleFillerGPU.h
        ReverseNonequilibriumShearFlowGPU.cuh
        ReverseNonequilibriumShearFlowGPU.h
        SorterGPU.cuh
        SorterGPU.h
        SRDCollisionMethodGPU.cuh
        SRDCollisionMethodGPU.h
        )
endif()

# add cc and h file for geometries
foreach(_geometry ${_geometries})
    # append force itself first
    list(APPEND _mpcd_sources ${_geometry}.cc)
    list(APPEND _mpcd_headers ${_geometry}.h)
endforeach()

# add cc and h file for forces
foreach(_force ${_forces})
    list(APPEND _mpcd_sources ${_force}.cc)
    list(APPEND _mpcd_headers ${_force}.h)
endforeach()

# generate cc and cu templates for forces and streaming geometries
foreach(_force ${_forces})
    # bulk geometry is special
    configure_file(
        BulkStreamingMethod.cc.inc
        BulkStreamingMethod${_force}.cc
        @ONLY
        )
    list(APPEND _mpcd_sources BulkStreamingMethod${_force}.cc)
    if(ENABLE_GPU)
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
        list(APPEND _mpcd_sources
            BulkStreamingMethod${_force}GPU.cc
            BulkStreamingMethod${_force}GPU.cu
            )
    endif()

    # then the bounce back geometries
    foreach(_geometry ${_geometries})
        configure_file(
            BounceBackStreamingMethod.cc.inc
            BounceBackStreamingMethod${_geometry}${_force}.cc
            @ONLY
            )
        list(APPEND _mpcd_sources BounceBackStreamingMethod${_geometry}${_force}.cc)
        if(ENABLE_GPU)
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
            list(APPEND _mpcd_sources
                BounceBackStreamingMethod${_geometry}${_force}GPU.cc
                BounceBackStreamingMethod${_geometry}${_force}GPU.cu
            )
        endif()
    endforeach()
endforeach()

# BounceBackNVE for the different geometries (no force involved)
foreach(_geometry ${_geometries})
    configure_file(
        BounceBackNVE.cc.inc
        BounceBackNVE${_geometry}.cc
        @ONLY
        )
    list(APPEND _mpcd_sources BounceBackNVE${_geometry}.cc)
    if(ENABLE_GPU)
        configure_file(
            BounceBackNVEGPU.cc.inc
            BounceBackNVE${_geometry}GPU.cc
            @ONLY
        )
        configure_file(
            BounceBackNVEGPU.cu.inc
            BounceBackNVE${_geometry}GPU.cu
            @ONLY
        )
        list(APPEND _mpcd_sources
            BounceBackNVE${_geometry}GPU.cc
            BounceBackNVE${_geometry}GPU.cu
        )
    endif()
endforeach()

# virtual particle fillers for the different geometries
foreach(_geometry ${_geometries})
    # skip fillers that are already done manually, guessed by filename
    if(${_geometry}Filler.cc IN_LIST _mpcd_sources)
        continue()
    endif()

    configure_file(
        RejectionVirtualParticleFiller.cc.inc
        RejectionVirtualParticleFiller${_geometry}.cc
        @ONLY
        )
    list(APPEND _mpcd_sources RejectionVirtualParticleFiller${_geometry}.cc)
    if(ENABLE_GPU)
        configure_file(
            RejectionVirtualParticleFillerGPU.cc.inc
            RejectionVirtualParticleFiller${_geometry}GPU.cc
            @ONLY
        )
        configure_file(
            RejectionVirtualParticleFillerGPUDrawKernel.cu.inc
            RejectionVirtualParticleFiller${_geometry}GPUDrawKernel.cu
            @ONLY
        )
        list(APPEND _mpcd_sources
            RejectionVirtualParticleFiller${_geometry}GPU.cc
            RejectionVirtualParticleFiller${_geometry}GPUDrawKernel.cu
        )
    endif()
endforeach()

hoomd_add_module(_mpcd SHARED ${_mpcd_sources} ${_mpcd_headers} NO_EXTRAS)
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
    update.py
    )
install(FILES ${files} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd)
copy_files_to_build("${files}" "mpcd" "*.py")

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
add_subdirectory(pytest)
