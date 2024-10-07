

set(_hpmc_gpu_shapes ShapeSphere
                 ShapeConvexPolygon
                 ShapeSimplePolygon
                 ShapeSpheropolygon
                 ShapePolyhedron
                 ShapeEllipsoid
                 ShapeFacetedEllipsoid
                 ShapeConvexPolyhedron
                 ShapeSpheropolyhedron
                 )

option(ENABLE_HPMC_SPHINX_GPU "Enable sphinx on the GPU" OFF)
if (ENABLE_HPMC_SPHINX_GPU)
    set(_hpmc_gpu_shapes ${_hpmc_gpu_shapes} ShapeSphinx)
endif()

set(_hpmc_gpu_union_shapes ShapeSphere
                       ShapeFacetedEllipsoid
                       ShapeSpheropolyhedron
   )

set(_hpmc_sources   module.cc
                    module_sphere.cc
                    module_convex_polygon.cc
                    module_simple_polygon.cc
                    module_spheropolygon.cc
                    module_polyhedron.cc
                    module_ellipsoid.cc
                    module_faceted_ellipsoid.cc
                    module_sphinx.cc
                    module_union_convex_polyhedron.cc
                    module_union_faceted_ellipsoid.cc
                    module_union_sphere.cc
                    module_convex_polyhedron.cc
                    module_convex_spheropolyhedron.cc
                    ExternalFieldWall.cc
                    ExternalPotential.cc
                    ExternalPotentialLinear.cc
                    PairPotential.cc
                    PairPotentialLennardJones.cc
                    PairPotentialExpandedGaussian.cc
                    PairPotentialLJGauss.cc
                    PairPotentialOPP.cc
                    PairPotentialStep.cc
                    PairPotentialUnion.cc
                    PairPotentialAngularStep.cc
                    ShapeUtils.cc
                    UpdaterBoxMC.cc
                    UpdaterQuickCompress.cc
                    IntegratorHPMC.cc
                    )

set(_hpmc_headers
    ComputeFreeVolumeGPU.cuh
    ComputeFreeVolumeGPU.h
    ComputeFreeVolume.h
    ComputeSDF.h
    ExternalField.h
    ExternalFieldHarmonic.h
    ExternalFieldWall.h
    ExternalPotential.h
    ExternalPotentialLinear.h
    GSDHPMCSchema.h
    GPUHelpers.cuh
    GPUTree.h
    HPMCCounters.h
    HPMCMiscFunctions.h
    IntegratorHPMC.h
    IntegratorHPMCMonoGPU.cuh
    IntegratorHPMCMonoGPUMoves.cuh
    IntegratorHPMCMonoGPUTypes.cuh
    IntegratorHPMCMonoGPUDepletants.cuh
    IntegratorHPMCMonoGPUDepletantsTypes.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryPhase1.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryPhase2.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryTypes.cuh
    IntegratorHPMCMonoGPU.h
    IntegratorHPMCMonoNEC.h
    IntegratorHPMCMono.h
    MinkowskiMath.h
    modules.h
    Moves.h
    OBB.h
    OBBTree.h
    PairPotential.h
    PairPotentialLennardJones.h
    PairPotentialExpandedGaussian.h
    PairPotentialLJGauss.h
    PairPotentialOPP.h
    PairPotentialStep.h
    PairPotentialUnion.h
    PairPotentialAngularStep.h
    ShapeConvexPolygon.h
    ShapeConvexPolyhedron.h
    ShapeEllipsoid.h
    ShapeFacetedEllipsoid.h
    ShapeMoves.h
    ShapePolyhedron.h
    ShapeSimplePolygon.h
    ShapeSphere.h
    ShapeSpheropolygon.h
    ShapeSpheropolyhedron.h
    ShapeSphinx.h
    ShapeUnion.h
    ShapeUtils.h
    SphinxOverlap.h
    UpdaterBoxMC.h
    UpdaterClusters.h
    UpdaterClustersGPU.cuh
    UpdaterClustersGPU.h
    UpdaterClustersGPUDepletants.cuh
    UpdaterMuVT.h
    UpdaterQuickCompress.h
    UpdaterShape.h
    XenoCollide2D.h
    XenoCollide3D.h
    XenoSweep3D.h
    )

set(_hpmc_cu_sources IntegratorHPMCMonoGPU.cu
                     IntegratorHPMCMonoGPUDepletants.cu
                     UpdaterClustersGPU.cu
                     )

set(_hpmc_kernel_templates kernel_free_volume
                           kernel_gen_moves
                           kernel_narrow_phase
                           kernel_insert_depletants
                           kernel_update_pdata
                           kernel_cluster_overlaps
                           kernel_cluster_depletants
                           kernel_cluster_transform
                           kernel_depletants_auxilliary_phase1
                           kernel_depletants_auxilliary_phase2)

if(ENABLE_HIP)
    # expand the shape x GPU kernel matrix of template instantiations
    foreach(KERNEL ${_hpmc_kernel_templates})
        foreach(SHAPE ${_hpmc_gpu_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(IS_UNION_SHAPE FALSE)
            set(_kernel_cu ${KERNEL}_${SHAPE}.cu)
            configure_file(${KERNEL}.cu.inc ${_kernel_cu} @ONLY)
            set(_hpmc_cu_sources ${_hpmc_cu_sources} ${_kernel_cu})
        endforeach()

        foreach(SHAPE ${_hpmc_gpu_union_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(_kernel_cu ${KERNEL}_union_${SHAPE}.cu)
            set(IS_UNION_SHAPE TRUE)
            configure_file(${KERNEL}.cu.inc ${_kernel_cu} @ONLY)
            set(_hpmc_cu_sources ${_hpmc_cu_sources} ${_kernel_cu})
        endforeach()
    endforeach()
endif(ENABLE_HIP)

if (ENABLE_HIP)
set(_cuda_sources ${_hpmc_cu_sources})
set_source_files_properties(${_hpmc_cu_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif (ENABLE_HIP)

hoomd_add_module(_hpmc SHARED ${_hpmc_sources} ${_cuda_sources} ${_hpmc_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_hpmc ALIAS _hpmc)
if (APPLE)
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

if(ENABLE_HIP)
    # enable out-of-source build
    target_include_directories(_hpmc PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
endif()

# link the library to its dependencies
target_link_libraries(_hpmc PUBLIC _hoomd)

if (ENABLE_HIP AND HIP_PLATFORM STREQUAL "nvcc")
target_link_libraries(_hpmc PUBLIC CUDA::cusparse )
endif()

# install the library
install(TARGETS _hpmc EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files   compute.py
            __init__.py
            integrate.py
            update.py
            shape_move.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc
       )

copy_files_to_build("${files}" "hpmc" "*.py")

# install headers in installation target
install(FILES ${_hpmc_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/hpmc
       )

add_subdirectory(pytest)
add_subdirectory(tune)
add_subdirectory(pair)
add_subdirectory(external)
add_subdirectory(nec)

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
