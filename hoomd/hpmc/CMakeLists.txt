# Maintainer: joaander

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
                    module_external_field.cc
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
                    UpdaterBoxMC.cc
                    IntegratorHPMC.cc
                    )

set(_hpmc_headers
    AnalyzerSDF.h
    ComputeFreeVolumeGPU.cuh
    ComputeFreeVolumeGPU.h
    ComputeFreeVolume.h
    ExternalFieldComposite.h
    ExternalField.h
    ExternalFieldLattice.h
    ExternalFieldWall.h
    GSDHPMCSchema.h
    GPUHelpers.cuh
    GPUTree.h
    HPMCCounters.h
    HPMCMiscFunctions.h
    HPMCPrecisionSetup.h
    IntegratorHPMC.h
    IntegratorHPMCMonoGPU.cuh
    IntegratorHPMCMonoGPUMoves.cuh
    IntegratorHPMCMonoGPUTypes.cuh
    IntegratorHPMCMonoGPUHelpers.cuh
    IntegratorHPMCMonoGPUDepletants.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryHelpers.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryPhase1.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryPhase2.cuh
    IntegratorHPMCMonoGPUDepletantsAuxilliaryTypes.cuh
    IntegratorHPMCMonoGPUJIT.inc
    IntegratorHPMCMonoGPU.h
    IntegratorHPMCMono.h
    MinkowskiMath.h
    modules.h
    Moves.h
    OBB.h
    OBBTree.h
    ShapeConvexPolygon.h
    ShapeConvexPolyhedron.h
    ShapeEllipsoid.h
    ShapeFacetedEllipsoid.h
    ShapePolyhedron.h
    ShapeProxy.h
    ShapeSimplePolygon.h
    ShapeSphere.h
    ShapeSpheropolygon.h
    ShapeSpheropolyhedron.h
    ShapeSphinx.h
    ShapeUnion.h
    SphinxOverlap.h
    UpdaterClusters.h
    UpdaterClustersGPU.cuh
    UpdaterClustersGPUDepletants.cuh
    UpdaterExternalFieldWall.h
    UpdaterMuVT.h
    UpdaterRemoveDrift.h
    XenoCollide2D.h
    XenoCollide3D.h
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
            configure_file(${KERNEL}.cu.in ${_kernel_cu} @ONLY)
            set(_hpmc_cu_sources ${_hpmc_cu_sources} ${_kernel_cu})
        endforeach()

        foreach(SHAPE ${_hpmc_gpu_union_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(_kernel_cu ${KERNEL}_union_${SHAPE}.cu)
            set(IS_UNION_SHAPE TRUE)
            configure_file(${KERNEL}.cu.in ${_kernel_cu} @ONLY)
            set(_hpmc_cu_sources ${_hpmc_cu_sources} ${_kernel_cu})
        endforeach()
    endforeach()
endif(ENABLE_HIP)

if (ENABLE_HIP)
set(_cuda_sources ${_hpmc_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_hpmc SHARED ${_hpmc_sources} ${_cuda_sources} ${_hpmc_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_hpmc ALIAS _hpmc)
if (APPLE)
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN")
endif()

if(ENABLE_HIP)
    # enable out-of-source build
    target_include_directories(_hpmc PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
endif()

# link the library to its dependencies
target_link_libraries(_hpmc PUBLIC _hoomd)

if (ENABLE_HIP AND HIP_PLATFORM STREQUAL "nvcc")
target_link_libraries(_hpmc PUBLIC CUDA::cusparse )
endif()

fix_cudart_rpath(_hpmc)

# install the library
install(TARGETS _hpmc EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc
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
        COMMENT    "Copy hoomd/hpmc/${file}"
    )
ENDMACRO(copy_file)

set(files   analyze.py
            compute.py
            data.py
            __init__.py
            integrate.py
            update.py
            util.py
            field.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hpmc ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_hpmc_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/hpmc
       )

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

if (BUILD_VALIDATION)
    add_subdirectory(validation)
endif()
