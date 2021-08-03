# Maintainer: joaander

set(_md_sources module-md.cc
                   ActiveForceCompute.cc
                   BondTablePotential.cc
                   CommunicatorGrid.cc
                   ConstExternalFieldDipoleForceCompute.cc
                   ConstraintEllipsoid.cc
                   ConstraintSphere.cc
                   CosineSqAngleForceCompute.cc
                   OneDConstraint.cc
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
                   MuellerPlatheFlow.cc
                   )

set(_md_headers ActiveForceComputeGPU.h
                ActiveForceCompute.h
                AllAnisoPairPotentials.h
                AllBondPotentials.h
                AllExternalPotentials.h
                AllPairPotentials.h
                AllSpecialPairPotentials.h
                AllTripletPotentials.h
                AnisoPotentialPairGPU.cuh
                AnisoPotentialPairGPU.h
                AnisoPotentialPair.h
                BondTablePotentialGPU.h
                BondTablePotential.h
                CommunicatorGridGPU.h
                CommunicatorGrid.h
                ConstExternalFieldDipoleForceCompute.h
                ConstraintEllipsoidGPU.h
                ConstraintEllipsoid.h
                ConstraintSphereGPU.h
                ConstraintSphere.h
                CosineSqAngleForceComputeGPU.h
                CosineSqAngleForceCompute.h
                Enforce2DUpdaterGPU.h
                Enforce2DUpdater.h
                EvaluatorBondFENE.h
                EvaluatorBondHarmonic.h
                EvaluatorSpecialPairLJ.h
                EvaluatorSpecialPairCoulomb.h
                EvaluatorConstraintEllipsoid.h
                EvaluatorConstraint.h
                EvaluatorConstraintSphere.h
                EvaluatorExternalElectricField.h
                EvaluatorExternalPeriodic.h
                EvaluatorPairBuckingham.h
                EvaluatorPairDipole.h
                EvaluatorPairDPDLJThermo.h
                EvaluatorPairDPDThermo.h
                EvaluatorPairEwald.h
                EvaluatorPairForceShiftedLJ.h
                EvaluatorPairGauss.h
                EvaluatorPairGB.h
                EvaluatorPairLJ.h
                EvaluatorPairLJ1208.h
                EvaluatorPairMie.h
                EvaluatorPairMoliere.h
                EvaluatorPairMorse.h
                EvaluatorPairFourier.h
                EvaluatorPairReactionField.h
                EvaluatorPairSLJ.h
                EvaluatorPairYukawa.h
                EvaluatorPairZBL.h
                EvaluatorTersoff.h
                EvaluatorWalls.h
                FIREEnergyMinimizerGPU.h
                FIREEnergyMinimizer.h
                ForceCompositeGPU.h
                ForceComposite.h
                ForceDistanceConstraintGPU.h
                ForceDistanceConstraint.h
                HarmonicAngleForceComputeGPU.h
                HarmonicAngleForceCompute.h
                HarmonicDihedralForceComputeGPU.h
                HarmonicDihedralForceCompute.h
                HarmonicImproperForceComputeGPU.h
                HarmonicImproperForceCompute.h
                IntegrationMethodTwoStep.h
                IntegratorTwoStep.h
                MolecularForceCompute.cuh
                MolecularForceCompute.h
                NeighborListBinned.h
                NeighborListGPUBinned.h
                NeighborListGPU.h
                NeighborListGPUStencil.h
                NeighborListGPUTree.h
                NeighborList.h
                NeighborListStencil.h
                NeighborListTree.h
                OPLSDihedralForceComputeGPU.h
                OPLSDihedralForceCompute.h
                PotentialBondGPU.h
                PotentialBondGPU.cuh
                PotentialBond.h
                PotentialExternalGPU.h
                PotentialExternalGPU.cuh
                PotentialExternal.h
                PotentialPairDPDThermoGPU.h
                PotentialPairDPDThermoGPU.cuh
                PotentialPairDPDThermo.h
                PotentialPairGPU.h
                PotentialPairGPU.cuh
                PotentialPair.h
                PotentialSpecialPairGPU.h
                PotentialSpecialPair.h
                PotentialTersoffGPU.h
                PotentialTersoff.h
                PPPMForceComputeGPU.h
                PPPMForceCompute.h
                QuaternionMath.h
                TableAngleForceComputeGPU.h
                TableAngleForceCompute.h
                TableDihedralForceComputeGPU.h
                TableDihedralForceCompute.h
                TablePotentialGPU.h
                TablePotential.h
                TempRescaleUpdater.h
                TwoStepBDGPU.h
                TwoStepBD.h
                TwoStepBerendsenGPU.h
                TwoStepBerendsen.h
                TwoStepLangevinBase.h
                TwoStepLangevinGPU.h
                TwoStepLangevin.h
                TwoStepNPTMTKGPU.h
                TwoStepNPTMTK.h
                TwoStepNVEGPU.h
                TwoStepNVE.h
                TwoStepNVTMTKGPU.h
                TwoStepNVTMTK.h
                WallData.h
                ZeroMomentumUpdater.h
                )

if (ENABLE_CUDA)
list(APPEND _md_sources ActiveForceComputeGPU.cc
                           BondTablePotentialGPU.cc
                           CommunicatorGridGPU.cc
                           ConstraintEllipsoidGPU.cc
                           ConstraintSphereGPU.cc
                           OneDConstraintGPU.cc
                           Enforce2DUpdaterGPU.cc
                           FIREEnergyMinimizerGPU.cc
                           ForceCompositeGPU.cc
                           ForceDistanceConstraintGPU.cc
                           HarmonicAngleForceComputeGPU.cc
                           HarmonicDihedralForceComputeGPU.cc
                           HarmonicImproperForceComputeGPU.cc
                           MolecularForceCompute.cu
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
                           MuellerPlatheFlowGPU.cc
                           CosineSqAngleForceComputeGPU.cc
                           )
endif()

set(_md_cu_sources ActiveForceComputeGPU.cu
                      AllDriverAnisoPotentialPairGPU.cu
                      AllDriverPotentialBondGPU.cu
                      AllDriverPotentialSpecialPairGPU.cu
                      BuckinghamDriverPotentialPairGPU.cu
                      DLVODriverPotentialPairGPU.cu
                      DPDLJThermoDriverPotentialPairGPU.cu
                      DPDThermoDriverPotentialPairGPU.cu
                      EwaldDriverPotentialPairGPU.cu
                      ForceShiftedLJDriverPotentialPairGPU.cu
                      GaussDriverPotentialPairGPU.cu
                      LJDriverPotentialPairGPU.cu
                      MieDriverPotentialPairGPU.cu
                      MoliereDriverPotentialPairGPU.cu
                      MorseDriverPotentialPairGPU.cu
                      FourierDriverPotentialPairGPU.cu
                      PairLJ1208DriverPotentialPairGPU.cu
                      ReactionFieldDriverPotentialPairGPU.cu
                      SLJDriverPotentialPairGPU.cu
                      YukawaDriverPotentialPairGPU.cu
                      ZBLDriverPotentialPairGPU.cu
                      BondTablePotentialGPU.cu
                      CommunicatorGridGPU.cu
                      ConstraintEllipsoidGPU.cu
                      ConstraintSphereGPU.cu
                      OneDConstraintGPU.cu
                      DriverTersoffGPU.cu
                      Enforce2DUpdaterGPU.cu
                      FIREEnergyMinimizerGPU.cu
                      ForceCompositeGPU.cu
                      ForceDistanceConstraintGPU.cu
                      HarmonicAngleForceGPU.cu
                      HarmonicDihedralForceGPU.cu
                      HarmonicImproperForceGPU.cu
                      MolecularForceCompute.cu
                      NeighborListGPUBinned.cu
                      NeighborListGPU.cu
                      NeighborListGPUStencil.cu
                      NeighborListGPUTree.cu
                      OPLSDihedralForceGPU.cu
                      PotentialExternalGPU.cu
                      PPPMForceComputeGPU.cu
                      TableAngleForceGPU.cu
                      TableDihedralForceGPU.cu
                      TablePotentialGPU.cu
                      TwoStepBDGPU.cu
                      TwoStepBerendsenGPU.cu
                      TwoStepLangevinGPU.cu
                      TwoStepNPTMTKGPU.cu
                      TwoStepNVEGPU.cu
                      TwoStepNVTMTKGPU.cu
                      MuellerPlatheFlowGPU.cu
                      CosineSqAngleForceGPU.cu
                      )

if (THRUST_FIX)
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX THRUST_CUB_WRAPPED_NAMESPACE) 
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX HOOMD_THRUST)
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX HOOMD_CUB)  
  list(APPEND CUDA_NVCC_FLAGS "-DTHRUST_CUB_WRAPPED_NAMESPACE=THRUST_CUB_MD")
  list(APPEND CUDA_NVCC_FLAGS "-DHOOMD_THRUST=THRUST_CUB_MD::thrust")
  list(APPEND CUDA_NVCC_FLAGS "-DHOOMD_CUB=THRUST_CUB_MD::cub")      
endif()
  
if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_md_cu_sources} ${DFFT_CU_SOURCES} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module(_md SHARED ${_md_sources} ${_CUDA_GENERATED_FILES} ${DFFT_SOURCES} NO_EXTRAS)
if(APPLE)
set_target_properties(_md PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_md PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN")
endif()

# link the library to its dependencies
if (CUSOLVER_AVAILABLE)
    # CUDA 8.0 requires that we link in gomp
    target_link_libraries(_md PRIVATE _hoomd ${HOOMD_COMMON_LIBS} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} gomp)
else()
    target_link_libraries(_md PRIVATE _hoomd ${HOOMD_COMMON_LIBS})
endif()
if (ENABLE_CUDA)
    target_link_libraries(_md PRIVATE neighbor)
endif()

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

fix_cudart_rpath(_md)

# install the library
install(TARGETS _md
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/md
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
          integrate.py
          nlist.py
          pair.py
          update.py
          wall.py
          special_pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/md
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_md ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/md/${file}
        COMMENT    "Copy hoomd/include/hoomd/md/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_md_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/md
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_md_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_md_headers ALL DEPENDS ${_md_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

if (BUILD_VALIDATION)
    add_subdirectory(validation)
endif()
