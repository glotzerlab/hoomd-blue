# Maintainer: joaander

set(_md_sources ActiveForceCompute.cc
                   BondTablePotential.cc
                   CGCMMAngleForceCompute.cc
                   CGCMMForceCompute.cc
                   ConstExternalFieldDipoleForceCompute.cc
                   ConstraintEllipsoid.cc
                   ConstraintSphere.cc
                   EAMForceCompute.cc
                   Enforce2DUpdater.cc
                   FIREEnergyMinimizer.cc
                   ForceComposite.cc
                   ForceDistanceConstraint.cc
                   HarmonicAngleForceCompute.cc
                   HarmonicDihedralForceCompute.cc
                   HarmonicImproperForceCompute.cc
                   IntegrationMethodTwoStep.cc
                   IntegratorTwoStep.cc
                   MolecularForceCompute.cc
                   NeighborListBinned.cc
                   NeighborList.cc
                   NeighborListStencil.cc
                   NeighborListTree.cc
                   OPLSDihedralForceCompute.cc
                   PPPMForceCompute.cc
                   TableAngleForceCompute.cc
                   TableDihedralForceCompute.cc
                   TablePotential.cc
                   TempRescaleUpdater.cc
                   TwoStepBD.cc
                   TwoStepBerendsen.cc
                   TwoStepLangevinBase.cc
                   TwoStepLangevin.cc
                   TwoStepNPTMTK.cc
                   TwoStepNVE.cc
                   TwoStepNVTMTK.cc
                   ZeroMomentumUpdater.cc
                   )

if (ENABLE_CUDA)
list(APPEND _md_sources ActiveForceComputeGPU.cc
                           BondTablePotentialGPU.cc
                           CGCMMAngleForceComputeGPU.cc
                           CGCMMForceComputeGPU.cc
                           ConstraintEllipsoidGPU.cc
                           ConstraintSphereGPU.cc
                           EAMForceComputeGPU.cc
                           Enforce2DUpdaterGPU.cc
                           FIREEnergyMinimizerGPU.cc
                           ForceCompositeGPU.cc
                           ForceDistanceConstraintGPU.cc
                           HarmonicAngleForceComputeGPU.cc
                           HarmonicDihedralForceComputeGPU.cc
                           HarmonicImproperForceComputeGPU.cc
                           NeighborListGPU.cc
                           NeighborListGPUBinned.cc
                           NeighborListGPUStencil.cc
                           NeighborListGPUTree.cc
                           OPLSDihedralForceComputeGPU.cc
                           PPPMForceComputeGPU.cc
                           TableAngleForceComputeGPU.cc
                           TableDihedralForceComputeGPU.cc
                           TablePotentialGPU.cc
                           TwoStepBDGPU.cc
                           TwoStepBerendsenGPU.cc
                           TwoStepLangevinGPU.cc
                           TwoStepNPTMTKGPU.cc
                           TwoStepNVEGPU.cc
                           TwoStepNVTMTKGPU.cc
                           )
endif()

set(_md_cu_sources ActiveForceComputeGPU.cu
                      AllDriverAnisoPotentialPairGPU.cu
                      AllDriverPotentialBondGPU.cu
                      AllDriverPotentialPairGPU.cu
                      BondTablePotentialGPU.cu
                      CGCMMAngleForceGPU.cu
                      CGCMMForceGPU.cu
                      ConstraintEllipsoidGPU.cu
                      ConstraintSphereGPU.cu
                      DriverTersoffGPU.cu
                      EAMForceGPU.cu
                      Enforce2DUpdaterGPU.cu
                      FIREEnergyMinimizerGPU.cu
                      ForceCompositeGPU.cu
                      ForceDistanceConstraintGPU.cu
                      HarmonicAngleForceGPU.cu
                      HarmonicDihedralForceGPU.cu
                      HarmonicImproperForceGPU.cu
                      NeighborListGPUBinned.cu
                      NeighborListGPU.cu
                      NeighborListGPUStencil.cu
                      NeighborListGPUTree.cu
                      OPLSDihedralForceGPU.cu
                      PotentialExternalGPU.cu
                      PPPMForceGPU.cu
                      TableAngleForceGPU.cu
                      TableDihedralForceGPU.cu
                      TablePotentialGPU.cu
                      TwoStepBDGPU.cu
                      TwoStepBerendsenGPU.cu
                      TwoStepLangevinGPU.cu
                      TwoStepNPTMTKGPU.cu
                      TwoStepNVEGPU.cu
                      TwoStepNVTMTKGPU.cu
                      )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_md_sources} ${_md_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_md_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_md SHARED module-md.cc ${_md_sources} ${_CUDA_GENERATED_FILES})

# link the library to its dependencies
target_link_libraries(_md ${HOOMD_COMMON_LIBS})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_md PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_md PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_md PROPERTIES PREFIX "" OUTPUT_NAME "_md")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_md PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_md)

# install the library
# install(TARGETS _hoomd
#        LIBRARY DESTINATION ${_hoomd_so_install_dir}
#        ARCHIVE DESTINATION ${LIB_BASE_INSTALL_DIR}
#        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/md/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          angle.py
          bond.py
          charge.py
          constrain.py
          dihedral.py
          external.py
          force.py
          improper.py
          nlist.py
          pair.py
          wall.py
    )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_md ALL DEPENDS ${files})

if (BUILD_TESTING)
    # add_subdirectory(test-py)
    add_subdirectory(test)
endif()
