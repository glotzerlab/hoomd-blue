

set(_md_sources module-md.cc
                   ActiveForceCompute.cc
                   ActiveRotationalDiffusionUpdater.cc
                   AnisoPotentialPairALJ2.cc
                   AnisoPotentialPairALJ3.cc
                   AnisoPotentialPairDipole.cc
                   AnisoPotentialPairGB.cc
                   BondTablePotential.cc
                   CommunicatorGrid.cc
                   ComputeThermo.cc
                   ComputeThermoHMA.cc
                   CosineSqAngleForceCompute.cc
                   CustomForceCompute.cc
                   EvaluatorWalls.cc
                   FIREEnergyMinimizer.cc
                   ForceComposite.cc
                   ForceDistanceConstraint.cc
                   HarmonicAngleForceCompute.cc
                   HarmonicDihedralForceCompute.cc
                   HarmonicImproperForceCompute.cc
                   IntegrationMethodTwoStep.cc
                   IntegratorTwoStep.cc
                   ManifoldZCylinder.cc
                   ManifoldDiamond.cc
                   ManifoldEllipsoid.cc
                   ManifoldGyroid.cc
                   ManifoldXYPlane.cc
                   ManifoldPrimitive.cc
                   ManifoldSphere.cc
                   MolecularForceCompute.cc
                   MuellerPlatheFlow.cc
                   NeighborListBinned.cc
                   NeighborList.cc
                   NeighborListStencil.cc
                   NeighborListTree.cc
                   OPLSDihedralForceCompute.cc
                   PPPMForceCompute.cc
                   TableAngleForceCompute.cc
                   TableDihedralForceCompute.cc
                   TwoStepBD.cc
                   TwoStepBerendsen.cc
                   TwoStepLangevinBase.cc
                   TwoStepLangevin.cc
                   TwoStepNPTMTK.cc
                   TwoStepNVE.cc
                   TwoStepNVTMTK.cc
                   TwoStepNVTAlchemy.cc
                   WallData.cc
                   ZeroMomentumUpdater.cc
                   )

set(_md_headers ActiveForceComputeGPU.h
                ActiveForceCompute.h
                ActiveForceConstraintCompute.h
                ActiveForceConstraintComputeGPU.h
                ActiveForceConstraintComputeGPU.cuh
                ActiveRotationalDiffusionUpdater.h
                AlchemyData.h
                AnisoPotentialPairALJ2GPUKernel.cuh
                AnisoPotentialPairALJ3GPUKernel.cuh
                AnisoPotentialPairDipoleGPUKernel.cuh
                AnisoPotentialPairALJ2.h
                AnisoPotentialPairALJ3.h
                AnisoPotentialPairDipole.h
                AnisoPotentialPairGB.h
                AnisoPotentialPairGBGPUKernel.cuh
                AllPairPotentials.h
                AnisoPotentialPairGPU.cuh
                AnisoPotentialPairGPU.h
                AnisoPotentialPair.h
                BondTablePotentialGPU.h
                BondTablePotential.h
                CommunicatorGridGPU.h
                CommunicatorGrid.h
                ComputeThermoGPU.cuh
                ComputeThermoGPU.h
                ComputeThermoHMAGPU.cuh
                ComputeThermoHMAGPU.h
                ComputeThermo.h
                ComputeThermoHMA.h
                ComputeThermoTypes.h
                ComputeThermoHMATypes.h
                CosineSqAngleForceComputeGPU.h
                CosineSqAngleForceCompute.h
                CustomForceCompute.h
                EvaluatorBondFENE.h
                EvaluatorBondHarmonic.h
                EvaluatorBondTether.h
                EvaluatorSpecialPairLJ.h
                EvaluatorSpecialPairCoulomb.h
                EvaluatorExternalElectricField.h
                EvaluatorExternalPeriodic.h
                EvaluatorPairALJ.h
                EvaluatorPairBuckingham.h
                EvaluatorPairDipole.h
                EvaluatorPairDLVO.h
                EvaluatorPairDPDLJThermo.h
                EvaluatorPairDPDThermo.h
                EvaluatorPairEwald.h
                EvaluatorPairForceShiftedLJ.h
                EvaluatorPairGauss.h
                EvaluatorPairGB.h
                EvaluatorPairLJ.h
                EvaluatorPairLJ1208.h
                EvaluatorPairLJ0804.h
                EvaluatorPairMie.h
                EvaluatorPairExpandedMie.h
                EvaluatorPairMoliere.h
                EvaluatorPairMorse.h
                EvaluatorPairOPP.h
                EvaluatorPairFourier.h
                EvaluatorPairReactionField.h
                EvaluatorPairExpandedLJ.h
                EvaluatorPairTable.h
                EvaluatorPairTWF.h
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
                ManifoldZCylinder.h
                ManifoldDiamond.h
                ManifoldEllipsoid.h
                ManifoldGyroid.h
                ManifoldXYPlane.h
                ManifoldPrimitive.h
                ManifoldSphere.h
                MolecularForceCompute.cuh
                MolecularForceCompute.h
                MuellerPlatheFlowEnum.h
                MuellerPlatheFlow.h
                MuellerPlatheFlowGPU.h
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
                PotentialPairAlchemical.h
                PotentialPairAlchemicalNormalized.h
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
                TableAngleForceComputeGPU.h
                TableAngleForceCompute.h
                TableDihedralForceComputeGPU.h
                TableDihedralForceCompute.h
                TwoStepBDGPU.h
                TwoStepRATTLEBDGPU.h
                TwoStepRATTLEBDGPU.cuh
                TwoStepBD.h
                TwoStepRATTLEBD.h
                TwoStepBerendsenGPU.h
                TwoStepBerendsen.h
                TwoStepLangevinBase.h
                TwoStepLangevinGPU.h
                TwoStepRATTLELangevinGPU.h
                TwoStepRATTLELangevinGPU.cuh
                TwoStepLangevin.h
                TwoStepRATTLELangevin.h
                TwoStepNPTMTKGPU.h
                TwoStepNPTMTK.h
                TwoStepRATTLENVEGPU.h
                TwoStepRATTLENVEGPU.cuh
                TwoStepRATTLENVE.h
                TwoStepNVEGPU.h
                TwoStepNVE.h
                TwoStepNVTMTKGPU.h
                TwoStepNVTMTK.h
                AlchemostatTwoStep.h
                TwoStepNVTAlchemy.h
                WallData.h
                ZeroMomentumUpdater.h
                )

if (ENABLE_HIP)
list(APPEND _md_sources ActiveForceComputeGPU.cc
                           AnisoPotentialPairALJ2GPU.cc
                           AnisoPotentialPairALJ3GPU.cc
                           AnisoPotentialPairDipoleGPU.cc
                           AnisoPotentialPairGBGPU.cc
                           BondTablePotentialGPU.cc
                           CommunicatorGridGPU.cc
                           ComputeThermoGPU.cc
                           ComputeThermoHMAGPU.cc
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
                      AnisoPotentialPairALJ2GPUKernel.cu
                      AnisoPotentialPairALJ3GPUKernel.cu
                      AnisoPotentialPairDipoleGPUKernel.cu
                      AnisoPotentialPairGBGPUKernel.cu
                      BuckinghamDriverPotentialPairGPU.cu
                      ComputeThermoGPU.cu
                      ComputeThermoHMAGPU.cu
                      DLVODriverPotentialPairGPU.cu
                      DPDLJThermoDriverPotentialPairGPU.cu
                      DPDThermoDriverPotentialPairGPU.cu
                      EwaldDriverPotentialPairGPU.cu
                      ForceShiftedLJDriverPotentialPairGPU.cu
                      GaussDriverPotentialPairGPU.cu
                      LJDriverPotentialPairGPU.cu
                      LJGaussDriverPotentialPairGPU.cu
                      MieDriverPotentialPairGPU.cu
		              ExpandedMieDriverPotentialPairGPU.cu
                      MoliereDriverPotentialPairGPU.cu
                      MorseDriverPotentialPairGPU.cu
                      OPPDriverPotentialPairGPU.cu
                      TableDriverPotentialPairGPU.cu
                      TWFDriverPotentialPairGPU.cu
                      FourierDriverPotentialPairGPU.cu
                      PairLJ1208DriverPotentialPairGPU.cu
                      PairLJ0804DriverPotentialPairGPU.cu
                      ReactionFieldDriverPotentialPairGPU.cu
                      ExpandedLJDriverPotentialPairGPU.cu
                      YukawaDriverPotentialPairGPU.cu
                      ZBLDriverPotentialPairGPU.cu
                      BondTablePotentialGPU.cu
                      CommunicatorGridGPU.cu
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
                      PPPMForceComputeGPU.cu
                      TableAngleForceGPU.cu
                      TableDihedralForceGPU.cu
                      TwoStepBDGPU.cu
                      TwoStepBerendsenGPU.cu
                      TwoStepLangevinGPU.cu
                      TwoStepRATTLELangevinGPU.cu
                      TwoStepNPTMTKGPU.cu
                      TwoStepNVEGPU.cu
                      TwoStepRATTLENVEGPU.cu
                      TwoStepNVTMTKGPU.cu
                      MuellerPlatheFlowGPU.cu
                      CosineSqAngleForceGPU.cu
                      )

if (ENABLE_HIP)
set(_cuda_sources ${_md_cu_sources} ${DFFT_CU_SOURCES})
endif (ENABLE_HIP)

# generate pybind11 export cc files
set(_manifolds Cylinder
               Diamond
               Ellipsoid
               Gyroid
               Plane
               Primitive
               Sphere)

set(_rattle_methods BD Langevin NVE)

foreach(_manifold ${_manifolds})
    set(_manifold_cpp ${_manifold})
    if (_manifold STREQUAL "Cylinder")
        set(_manifold_cpp "ZCylinder")
    elseif (_manifold STREQUAL "Plane")
        set(_manifold_cpp "XYPlane")
    endif()

    configure_file(export_ActiveForceConstraintCompute.cc.inc
                   export_ActiveForceConstraintCompute${_manifold}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_ActiveForceConstraintCompute${_manifold}.cc)

    if (ENABLE_HIP)
        configure_file(export_ActiveForceConstraintComputeGPU.cc.inc
                       export_ActiveForceConstraintCompute${_manifold}GPU.cc
                       @ONLY)
        configure_file(ActiveForceConstraintComputeGPU.cu.inc
                       ActiveForceConstraintCompute${_manifold}GPU.cu
                       @ONLY)
        configure_file(TwoStepRATTLEGPU.cu.inc
                       TwoStepRATTLE${_manifold}GPU.cu
                       @ONLY)
        set(_cuda_sources ${_cuda_sources}
            export_ActiveForceConstraintCompute${_manifold}GPU.cc
            ActiveForceConstraintCompute${_manifold}GPU.cu
            TwoStepRATTLE${_manifold}GPU.cu
            )
    endif()

    foreach (_method ${_rattle_methods})
        configure_file(export_TwoStepRATTLE.cc.inc
                    export_TwoStepRATTLE${_method}${_manifold}.cc
                    @ONLY)
        set(_md_sources ${_md_sources} export_TwoStepRATTLE${_method}${_manifold}.cc)

        if (ENABLE_HIP)
            configure_file(export_TwoStepRATTLEGPU.cc.inc
                        export_TwoStepRATTLE${_method}${_manifold}GPU.cc
                        @ONLY)
            set(_cuda_sources ${_cuda_sources} export_TwoStepRATTLE${_method}${_manifold}GPU.cc)
        endif()
    endforeach()

endforeach()

# generate pybind11 export cc files
set(_manifolds Cylinder
               Diamond
               Ellipsoid
               Gyroid
               Plane
               Primitive
               Sphere)

set(_bonds Harmonic FENE Tether)

foreach(_bond ${_bonds})
    configure_file(export_PotentialBond.cc.inc
                   export_PotentialBond${_bond}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialBond${_bond}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialBondGPU.cc.inc
                       export_PotentialBond${_bond}GPU.cc
                       @ONLY)
        configure_file(PotentialBondGPUKernel.cu.inc
                       PotentialBond${_bond}GPUKernel.cu
                       @ONLY)
        set(_cuda_sources ${_cuda_sources}
            export_PotentialBond${_bond}GPU.cc
            PotentialBond${_bond}GPUKernel.cu
            )
    endif()
endforeach()

set(_pairs LJ Coulomb)

foreach(_pair ${_pairs})
    configure_file(export_PotentialSpecialPair.cc.inc
                   export_PotentialSpecialPair${_pair}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialSpecialPair${_pair}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialSpecialPairGPU.cc.inc
                       export_PotentialSpecialPair${_pair}GPU.cc
                       @ONLY)
        configure_file(PotentialSpecialPairGPUKernel.cu.inc
                       PotentialSpecialPair${_pair}GPUKernel.cu
                       @ONLY)
        set(_cuda_sources ${_cuda_sources}
            export_PotentialSpecialPair${_pair}GPU.cc
            PotentialSpecialPair${_pair}GPUKernel.cu
            )
    endif()
endforeach()

set(_triplets Tersoff SquareDensity RevCross)

foreach(_evaluator ${_triplets})
    configure_file(export_PotentialTersoff.cc.inc
                   export_PotentialTersoff${_evaluator}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialTersoff${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialTersoffGPU.cc.inc
                       export_PotentialTersoff${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialTersoffGPUKernel.cu.inc
                       PotentialTersoff${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_cuda_sources ${_cuda_sources}
            export_PotentialTersoff${_evaluator}GPU.cc
            PotentialTersoff${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

set(_external_evaluators Periodic ElectricField)

foreach(_evaluator ${_external_evaluators})
    configure_file(export_PotentialExternal.cc.inc
                   export_PotentialExternal${_evaluator}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialExternal${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialExternalGPU.cc.inc
                       export_PotentialExternal${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialExternalGPUKernel.cu.inc
                       PotentialExternal${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_cuda_sources ${_cuda_sources}
            export_PotentialExternal${_evaluator}GPU.cc
            PotentialExternal${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

set(_wall_evaluators LJ Yukawa ForceShiftedLJ Mie Gauss Morse)

foreach(_evaluator ${_wall_evaluators})
    configure_file(export_PotentialExternalWall.cc.inc
                   export_PotentialExternalWall${_evaluator}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialExternalWall${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialExternalWallGPU.cc.inc
                       export_PotentialExternalWall${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialExternalWallGPUKernel.cu.inc
                       PotentialExternalWall${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_cuda_sources ${_cuda_sources}
            export_PotentialExternalWall${_evaluator}GPU.cc
            PotentialExternalWall${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

pybind11_add_module(_md SHARED ${_md_sources} ${_cuda_sources} ${DFFT_SOURCES} ${_md_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_md ALIAS _md)

if(APPLE)
set_target_properties(_md PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_md PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
if (CUSOLVER_AVAILABLE)
    # CUDA 8.0 requires that we link in gomp
    target_link_libraries(_md PUBLIC _hoomd CUDA::cusolver CUDA::cusparse gomp)
else()
    target_link_libraries(_md PUBLIC _hoomd)
endif()
if (ENABLE_HIP)
    target_link_libraries(_md PRIVATE neighbor)
endif()

fix_cudart_rpath(_md)

# install the library
install(TARGETS _md EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files __init__.py
          angle.py
          bond.py
          compute.py
          constrain.py
          dihedral.py
          force.py
          improper.py
          integrate.py
          manifold.py
          many_body.py
          nlist.py
          update.py
          special_pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md
       )

copy_files_to_build("${files}" "md" "*.py")

# install headers in installation target
install(FILES ${_md_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/md
       )

add_subdirectory(data)

add_subdirectory(pair)

add_subdirectory(methods)

add_subdirectory(long_range)

add_subdirectory(external)
add_subdirectory(minimize)
add_subdirectory(alchemy)

if (BUILD_TESTING)
    # add_subdirectory(test-py)
    add_subdirectory(test)
endif()

add_subdirectory(pytest)

if (BUILD_VALIDATION)
    # add_subdirectory(validation)
endif()
