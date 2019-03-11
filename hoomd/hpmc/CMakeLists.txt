# Maintainer: joaander

option(ENABLE_HPMC_SPHINX_GPU "Enable sphinx on the GPU" OFF)
if (ENABLE_HPMC_SPHINX_GPU)
    add_definitions(-DENABLE_SPHINX_GPU)
endif()

option(ENABLE_HPMC_REINSERT "Enable ntrial>0 with depletant_mode='circumsphere' on the GPU" OFF)
if (ENABLE_HPMC_REINSERT)
    add_definitions(-DENABLE_HPMC_REINSERT)
endif()

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
    GPUTree.h
    HPMCCounters.h
    HPMCPrecisionSetup.h
    IntegratorHPMC.h
    IntegratorHPMCMonoGPU.cuh
    IntegratorHPMCMonoGPU.h
    IntegratorHPMCMono.h
    IntegratorHPMCMonoImplicitGPU.h
    IntegratorHPMCMonoImplicit.h
    IntegratorHPMCMonoImplicitNewGPU.cuh
    IntegratorHPMCMonoImplicitNewGPU.h
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
    UpdaterExternalFieldWall.h
    UpdaterMuVT.h
    UpdaterMuVTImplicit.h
    UpdaterRemoveDrift.h
    XenoCollide2D.h
    XenoCollide3D.h
    )

# if (ENABLE_CUDA)
# list(APPEND _hpmc_sources
#                            )
# endif()

set(_hpmc_cu_sources IntegratorHPMCMonoGPU.cu
                     IntegratorHPMCMonoImplicitGPU.cu
                     all_kernels_sphere.cu
                     all_kernels_convex_polygon.cu
                     all_kernels_simple_polygon.cu
                     all_kernels_spheropolygon.cu
                     all_kernels_polyhedron.cu
                     all_kernels_ellipsoid.cu
                     all_kernels_faceted_ellipsoid.cu
                     all_kernels_sphinx.cu
                     all_kernels_union_spheropolyhedron.cu
                     all_kernels_union_faceted_ellipsoid.cu
                     all_kernels_union_sphere.cu
                     all_kernels_convex_polyhedron.cu
                     all_kernels_convex_spheropolyhedron.cu
                     )

# quiet some warnings locally on files we can't modify
if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties(IntegratorHPMCMonoImplicitGPU.cu PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
endif()

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_hpmc_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module (_hpmc SHARED ${_hpmc_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)
if (APPLE)
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_hpmc PRIVATE ${HOOMD_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_hpmc PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_hpmc PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

fix_cudart_rpath(_hpmc)

# install the library
install(TARGETS _hpmc
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/hpmc
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
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/hpmc
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hpmc ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/hpmc/${file}
        COMMENT    "Copy hoomd/include/hoomd/hpmc/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_hpmc_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/hpmc
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_hpmc_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_hpmc_headers ALL DEPENDS ${_hpmc_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

if (BUILD_VALIDATION)
    add_subdirectory(validation)
endif()
