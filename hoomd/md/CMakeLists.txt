

set(_md_sources module-md.cc
                   ActiveForceCompute.cc
                   ActiveRotationalDiffusionUpdater.cc
                   AreaConservationMeshForceCompute.cc
                   AlchemostatTwoStep.cc
                   AnisoPotentialPairALJ2.cc
                   AnisoPotentialPairALJ3.cc
                   AnisoPotentialPairDipole.cc
                   AnisoPotentialPairGB.cc
                   AlchemyData.cc
                   BendingRigidityMeshForceCompute.cc
                   BondTablePotential.cc
                   CommunicatorGrid.cc
                   ComputeThermo.cc
                   ComputeThermoHMA.cc
                   ConstantForceCompute.cc
                   CosineSqAngleForceCompute.cc
                   CustomForceCompute.cc
                   EvaluatorWalls.cc
                   FIREEnergyMinimizer.cc
                   ForceComposite.cc
                   ForceDistanceConstraint.cc
                   HalfStepHook.cc
                   HarmonicAngleForceCompute.cc
                   HarmonicDihedralForceCompute.cc
                   HarmonicImproperForceCompute.cc
                   HelfrichMeshForceCompute.cc
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
                   PeriodicImproperForceCompute.cc
                   TableAngleForceCompute.cc
                   TableDihedralForceCompute.cc
                   TriangleAreaConservationMeshForceCompute.cc
                   TwoStepBD.cc
                   TwoStepLangevinBase.cc
                   TwoStepLangevin.cc
                   TwoStepConstantVolume.cc
                   TwoStepConstantPressure.cc
                   Thermostat.cc
                   TwoStepNVTAlchemy.cc
                   WallData.cc
		   VolumeConservationMeshForceCompute.cc
                   ZeroMomentumUpdater.cc
                   )

set(_md_headers ActiveForceComputeGPU.h
                ActiveForceCompute.h
                ActiveForceConstraintCompute.h
                ActiveForceConstraintComputeGPU.h
                ActiveForceConstraintComputeGPU.cuh
                ActiveRotationalDiffusionUpdater.h
                AlchemyData.h
                AnisoPotentialPairGPU.cuh
                AnisoPotentialPairGPU.h
                AnisoPotentialPair.h
                AreaConservationMeshForceCompute.h
                AreaConservationMeshForceComputeGPU.h
                AreaConservationMeshForceComputeGPU.cuh
                BendingRigidityMeshForceComputeGPU.h
                BendingRigidityMeshForceCompute.h
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
                ConstantForceComputeGPU.h
                ConstantForceCompute.h
                CosineSqAngleForceComputeGPU.h
                CosineSqAngleForceCompute.h
                CustomForceCompute.h
                EvaluatorBondFENE.h
                EvaluatorBondHarmonic.h
                EvaluatorBondTether.h
                EvaluatorSpecialPairLJ.h
                EvaluatorSpecialPairCoulomb.h
                EvaluatorExternalElectricField.h
		EvaluatorExternalMagneticField.h
                EvaluatorExternalPeriodic.h
                EvaluatorPairALJ.h
                EvaluatorPairBuckingham.h
                EvaluatorPairDipole.h
                EvaluatorPairDLVO.h
                EvaluatorPairDPDThermoLJ.h
                EvaluatorPairDPDThermoDPD.h
                EvaluatorPairEwald.h
                EvaluatorPairForceShiftedLJ.h
                EvaluatorPairGauss.h
                EvaluatorPairExpandedGaussian.h
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
		HelfrichMeshForceComputeGPU.h
		HelfrichMeshForceCompute.h
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
                PeriodicImproper.h
                PeriodicImproperForceCompute.h
                PeriodicImproperForceComputeGPU.h
                PPPMForceComputeGPU.h
                PPPMForceCompute.h
                TableAngleForceComputeGPU.h
                TableAngleForceCompute.h
                TableDihedralForceComputeGPU.h
                TableDihedralForceCompute.h
                TriangleAreaConservationMeshForceCompute.h
                TriangleAreaConservationMeshForceComputeGPU.h
                TriangleAreaConservationMeshForceComputeGPU.cuh
                TwoStepBDGPU.h
                TwoStepRATTLEBDGPU.h
                TwoStepRATTLEBDGPU.cuh
                TwoStepBD.h
                TwoStepRATTLEBD.h
                TwoStepLangevinBase.h
                TwoStepLangevinGPU.h
                TwoStepRATTLELangevinGPU.h
                TwoStepRATTLELangevinGPU.cuh
                TwoStepLangevin.h
                TwoStepRATTLELangevin.h
                TwoStepRATTLENVEGPU.h
                TwoStepRATTLENVEGPU.cuh
                TwoStepRATTLENVE.h
                TwoStepConstantVolume.h
                TwoStepConstantVolumeGPU.h
                TwoStepConstantPressure.h
                AlchemostatTwoStep.h
                TwoStepNVTAlchemy.h
		VolumeConservationMeshForceCompute.h
		VolumeConservationMeshForceComputeGPU.h
                AlchemostatTwoStep.h
                WallData.h
                ZeroMomentumUpdater.h
                )

if (ENABLE_HIP)
list(APPEND _md_sources ActiveForceComputeGPU.cc
                           AreaConservationMeshForceComputeGPU.cc
                           AnisoPotentialPairALJ2GPU.cc
                           AnisoPotentialPairALJ3GPU.cc
                           AnisoPotentialPairDipoleGPU.cc
                           AnisoPotentialPairGBGPU.cc
                           BendingRigidityMeshForceComputeGPU.cc
                           BondTablePotentialGPU.cc
                           CommunicatorGridGPU.cc
                           ComputeThermoGPU.cc
                           ComputeThermoHMAGPU.cc
                           ConstantForceComputeGPU.cc
                           FIREEnergyMinimizerGPU.cc
                           ForceCompositeGPU.cc
                           ForceDistanceConstraintGPU.cc
                           HarmonicAngleForceComputeGPU.cc
                           HarmonicDihedralForceComputeGPU.cc
                           HarmonicImproperForceComputeGPU.cc
			   HelfrichMeshForceComputeGPU.cc
                           MolecularForceCompute.cu
                           NeighborListGPU.cc
                           NeighborListGPUBinned.cc
                           NeighborListGPUStencil.cc
                           NeighborListGPUTree.cc
                           OPLSDihedralForceComputeGPU.cc
                           PeriodicImproperForceComputeGPU.cc
                           PPPMForceComputeGPU.cc
                           TableAngleForceComputeGPU.cc
                           TableDihedralForceComputeGPU.cc
                           TriangleAreaConservationMeshForceComputeGPU.cc
                           TwoStepBDGPU.cc
                           TwoStepLangevinGPU.cc
			   VolumeConservationMeshForceComputeGPU.cc
                           TwoStepConstantVolumeGPU.cc
                           TwoStepConstantPressureGPU.cc
                           MuellerPlatheFlowGPU.cc
                           CosineSqAngleForceComputeGPU.cc
                           )
endif()

set(_md_cu_sources ActiveForceComputeGPU.cu
                      AreaConservationMeshForceComputeGPU.cu
                      AnisoPotentialPairALJ2GPUKernel.cu
                      AnisoPotentialPairALJ3GPUKernel.cu
                      AnisoPotentialPairDipoleGPUKernel.cu
                      AnisoPotentialPairGBGPUKernel.cu
		      BendingRigidityMeshForceComputeGPU.cu
                      ComputeThermoGPU.cu
                      ComputeThermoHMAGPU.cu
                      ConstantForceComputeGPU.cu
                      BondTablePotentialGPU.cu
                      CommunicatorGridGPU.cu
                      FIREEnergyMinimizerGPU.cu
                      ForceCompositeGPU.cu
                      ForceDistanceConstraintGPU.cu
                      HarmonicAngleForceGPU.cu
                      HarmonicDihedralForceGPU.cu
                      HarmonicImproperForceGPU.cu
		      HelfrichMeshForceComputeGPU.cu
                      MolecularForceCompute.cu
                      NeighborListGPUBinned.cu
                      NeighborListGPU.cu
                      NeighborListGPUStencil.cu
                      NeighborListGPUTree.cu
                      OPLSDihedralForceGPU.cu
                      PeriodicImproperForceGPU.cu
                      PPPMForceComputeGPU.cu
                      TableAngleForceGPU.cu
                      TableDihedralForceGPU.cu
                      TriangleAreaConservationMeshForceComputeGPU.cu
                      TwoStepBDGPU.cu
                      TwoStepLangevinGPU.cu
                      TwoStepRATTLELangevinGPU.cu
                      TwoStepConstantPressureGPU.cu
                      TwoStepConstantVolumeGPU.cu
                      TwoStepNVEGPU.cu
                      TwoStepRATTLENVEGPU.cu
		      VolumeConservationMeshForceComputeGPU.cu
                      MuellerPlatheFlowGPU.cu
                      CosineSqAngleForceGPU.cu
                      )

if (ENABLE_HIP)
set(_cuda_sources ${_md_cu_sources} ${DFFT_CU_SOURCES})
set_source_files_properties(${_md_cu_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
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
        set(_md_sources ${_md_sources} export_ActiveForceConstraintCompute${_manifold}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            ActiveForceConstraintCompute${_manifold}GPU.cu
            TwoStepRATTLE${_manifold}GPU.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
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
            set(_md_sources ${_md_sources} export_TwoStepRATTLE${_method}${_manifold}GPU.cc)
            set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
        endif()
    endforeach()

endforeach()

# generate pybind11 export cc files
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
        set(_md_sources ${_md_sources} export_PotentialBond${_bond}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialBond${_bond}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    endif()
endforeach()

foreach(_bond ${_bonds})
    configure_file(export_PotentialMeshBond.cc.inc
                   export_PotentialMeshBond${_bond}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialMeshBond${_bond}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialMeshBondGPU.cc.inc
                       export_PotentialMeshBond${_bond}GPU.cc
                       @ONLY)
        configure_file(PotentialMeshBondGPUKernel.cu.inc
                       PotentialMeshBond${_bond}GPUKernel.cu
                       @ONLY)
        set(_md_sources ${_md_sources} export_PotentialMeshBond${_bond}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialMeshBond${_bond}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
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
        set(_md_sources ${_md_sources} export_PotentialSpecialPair${_pair}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialSpecialPair${_pair}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
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
        set(_md_sources ${_md_sources} export_PotentialTersoff${_evaluator}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialTersoff${_evaluator}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    endif()
endforeach()

set(_external_evaluators Periodic ElectricField MagneticField)

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
        set(_md_sources ${_md_sources} export_PotentialExternal${_evaluator}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialExternal${_evaluator}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
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
        set(_md_sources ${_md_sources} export_PotentialExternalWall${_evaluator}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialExternalWall${_evaluator}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    endif()
endforeach()

set(_dpdthermo_evaluators DPD LJ)

foreach(_evaluator ${_dpdthermo_evaluators})
    configure_file(export_PotentialPairDPDThermo.cc.inc
                   export_PotentialPairDPDThermo${_evaluator}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialPairDPDThermo${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialPairDPDThermoGPU.cc.inc
                       export_PotentialPairDPDThermo${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialPairDPDThermoGPUKernel.cu.inc
                       PotentialPairDPDThermo${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_md_sources ${_md_sources} export_PotentialPairDPDThermo${_evaluator}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialPairDPDThermo${_evaluator}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    endif()
endforeach()

set(_pair_evaluators Buckingham
                     LJ
                     LJ1208
                     LJ0804
                     Gauss
                     ExpandedLJ
                     ExpandedMie
                     Yukawa
                     Ewald
                     Morse
                     ConservativeDPD
                     Moliere
                     ZBL
                     Mie
                     ReactionField
                     DLVO
                     Fourier
                     OPP
                     TWF
                     LJGauss
                     ForceShiftedLJ
                     Table
                     ExpandedGaussian)


foreach(_evaluator ${_pair_evaluators})
    set(_evaluator_cpp ${_evaluator})
    if (_evaluator STREQUAL "ConservativeDPD")
        set(_evaluator_cpp "DPDThermoDPD")
    endif()

    configure_file(export_PotentialPair.cc.inc
                   export_PotentialPair${_evaluator}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialPair${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialPairGPU.cc.inc
                       export_PotentialPair${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialPairGPUKernel.cu.inc
                       PotentialPair${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_md_sources ${_md_sources} export_PotentialPair${_evaluator}GPU.cc)
        set(_cuda_sources ${_cuda_sources}
            PotentialPair${_evaluator}GPUKernel.cu
            )
        set_source_files_properties(${_cuda_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
    endif()
endforeach()

set(_alchemical_pair_evaluators LJGauss)

foreach(_evaluator ${_alchemical_pair_evaluators})
    configure_file(export_PotentialPairAlchemical.cc.inc
                   export_PotentialPairAlchemical${_evaluator}.cc
                   @ONLY)
    set(_md_sources ${_md_sources} export_PotentialPairAlchemical${_evaluator}.cc)
endforeach()

hoomd_add_module(_md SHARED ${_md_sources} ${_cuda_sources} ${DFFT_SOURCES} ${_md_headers} NO_EXTRAS)
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
          half_step_hook.py
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

add_subdirectory(mesh)

add_subdirectory(long_range)

add_subdirectory(external)
add_subdirectory(minimize)
add_subdirectory(alchemy)
add_subdirectory(tune)

if (BUILD_TESTING)
    add_subdirectory(test)
endif()

add_subdirectory(pytest)
