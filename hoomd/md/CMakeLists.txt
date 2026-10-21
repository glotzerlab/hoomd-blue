set(_md_sources
    module-md.cc
    ActiveForceCompute.cc
    ActiveRotationalDiffusionUpdater.cc
    AreaConservationMeshForceCompute.cc
    AlchemostatTwoStep.cc
    AnisoPotentialPairALJ2.cc
    AnisoPotentialPairALJ3.cc
    AnisoPotentialPairDipole.cc
    AnisoPotentialPairYLZ.cc
    AnisoPotentialPairGB.cc
    AlchemyData.cc
    BendingRigidityMeshForceCompute.cc
    BondTablePotential.cc
    BoxDeformer.cc
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
    FrictionPairFrictionLJBase.cc
    HalfStepHook.cc
    HarmonicAngleForceCompute.cc
    HarmonicDihedralForceCompute.cc
    HarmonicImproperForceCompute.cc
    HelfrichMeshForceCompute.cc
    IntegrationMethodTwoStep.cc
    IntegratorTwoStep.cc
    LeesEdwardsBoxDeformer.cc
    ManifoldZCylinder.cc
    ManifoldDiamond.cc
    ManifoldEllipsoid.cc
    ManifoldGyroid.cc
    ManifoldXYPlane.cc
    ManifoldPrimitive.cc
    ManifoldSphere.cc
    MeshDynamicBondUpdater.cc
    MeshForceCompute.cc
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
    ${DFFT_SOURCES}
    )

set(_md_headers
    ActiveForceCompute.h
    ActiveForceConstraintCompute.h
    ActiveRotationalDiffusionUpdater.h
    AlchemostatTwoStep.h
    AlchemyData.h
    AnisoPotentialPair.h
    AreaConservationMeshParameters.h
    AreaConservationMeshForceCompute.h
    BendingRigidityMeshForceCompute.h
    BondTablePotential.h
    BoxDeformer.h
    CommunicatorGrid.h
    ComputeThermo.h
    ComputeThermoHMA.h
    ComputeThermoTypes.h
    ComputeThermoHMATypes.h
    ConstantForceCompute.h
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
    EvaluatorPairFrictionLJBase.h
    EvaluatorPairFrictionLJVariants.h
    EvaluatorPairYLZ.h
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
    EvaluatorPairWangFrenkel.h
    EvaluatorPairZBL.h
    EvaluatorPairZetterling.h
    EvaluatorTersoff.h
    EvaluatorWalls.h
    FIREEnergyMinimizer.h
    ForceComposite.h
    ForceDistanceConstraint.h
    PatchEnvelope.h
    HarmonicAngleForceCompute.h
    HarmonicDihedralForceCompute.h
    HarmonicImproperForceCompute.h
    HelfrichMeshForceCompute.h
    IntegrationMethodTwoStep.h
    IntegratorTwoStep.h
    LeesEdwardsBoxDeformer.h
    ManifoldZCylinder.h
    ManifoldDiamond.h
    ManifoldEllipsoid.h
    ManifoldGyroid.h
    ManifoldXYPlane.h
    ManifoldPrimitive.h
    ManifoldSphere.h
    MeshDynamicBondUpdater.h
    MeshForceCompute.h
    MolecularForceCompute.h
    MuellerPlatheFlowEnum.h
    MuellerPlatheFlow.h
    NeighborList.h
    NeighborListBinned.h
    NeighborListStencil.h
    NeighborListTree.h
    OPLSDihedralForceCompute.h
    PairModulator.h
    PotentialBond.h
    PotentialExternal.h
    PotentialPairAlchemical.h
    PotentialPairAlchemicalNormalized.h
    PotentialPairDPDThermo.h
    PotentialPair.h
    PotentialSpecialPair.h
    PotentialTersoff.h
    PeriodicImproper.h
    PeriodicImproperForceCompute.h
    PPPMForceCompute.h
    FrictionPair.h
    TableAngleForceCompute.h
    TableDihedralForceCompute.h
    TriangleAreaConservationMeshParameters.h
    TriangleAreaConservationMeshForceCompute.h
    TwoStepBD.h
    TwoStepConstantPressure.h
    TwoStepConstantVolume.h
    TwoStepLangevinBase.h
    TwoStepLangevin.h
    TwoStepRATTLEBD.h
    TwoStepRATTLELangevin.h
    TwoStepRATTLENVE.h
    TwoStepNVTAlchemy.h
    VolumeConservationMeshParameters.h
    VolumeConservationMeshForceCompute.h
    WallData.h
    ZeroMomentumUpdater.h
    )

if (ENABLE_GPU)
    list(APPEND _md_sources
        ActiveForceComputeGPU.cc
        ActiveForceComputeGPU.cu
        AnisoPotentialPairALJ2GPU.cc
        AnisoPotentialPairALJ2GPUKernel.cu
        AnisoPotentialPairALJ3GPU.cc
        AnisoPotentialPairALJ3GPUKernel.cu
        AnisoPotentialPairDipoleGPU.cc
        AnisoPotentialPairDipoleGPUKernel.cu
        AnisoPotentialPairYLZGPU.cc
        AnisoPotentialPairYLZGPUKernel.cu
        AnisoPotentialPairGBGPU.cc
        AnisoPotentialPairGBGPUKernel.cu
        AreaConservationMeshForceComputeGPU.cc
        AreaConservationMeshForceComputeGPU.cu
        BendingRigidityMeshForceComputeGPU.cc
        BendingRigidityMeshForceComputeGPU.cu
        BondTablePotentialGPU.cc
        BondTablePotentialGPU.cu
        BoxDeformerGPU.cu
        CommunicatorGridGPU.cc
        CommunicatorGridGPU.cu
        ComputeThermoGPU.cc
        ComputeThermoGPU.cu
        ComputeThermoHMAGPU.cc
        ComputeThermoHMAGPU.cu
        ConstantForceComputeGPU.cc
        ConstantForceComputeGPU.cu
        CosineSqAngleForceComputeGPU.cc
        CosineSqAngleForceGPU.cu
        FIREEnergyMinimizerGPU.cc
        FIREEnergyMinimizerGPU.cu
        ForceCompositeGPU.cc
        ForceCompositeGPU.cu
        ForceDistanceConstraintGPU.cc
        ForceDistanceConstraintGPU.cu
        HarmonicAngleForceComputeGPU.cc
        HarmonicAngleForceGPU.cu
        HarmonicDihedralForceComputeGPU.cc
        HarmonicDihedralForceGPU.cu
        HarmonicImproperForceComputeGPU.cc
        HarmonicImproperForceGPU.cu
        HelfrichMeshForceComputeGPU.cc
        HelfrichMeshForceComputeGPU.cu
        LeesEdwardsBoxDeformerGPU.cu
        MolecularForceCompute.cu
        MuellerPlatheFlowGPU.cc
        MuellerPlatheFlowGPU.cu
        NeighborListGPU.cc
        NeighborListGPU.cu
        NeighborListGPUBinned.cc
        NeighborListGPUBinned.cu
        NeighborListGPUStencil.cc
        NeighborListGPUStencil.cu
        NeighborListGPUTree.cc
        NeighborListGPUTree.cu
        OPLSDihedralForceComputeGPU.cc
        OPLSDihedralForceGPU.cu
        PeriodicImproperForceComputeGPU.cc
        PeriodicImproperForceGPU.cu
        PPPMForceComputeGPU.cc
        PPPMForceComputeGPU.cu
        FrictionPairFrictionLJBaseGPU.cc
        FrictionPairFrictionLJBaseGPUKernel.cu
        TableAngleForceComputeGPU.cc
        TableAngleForceGPU.cu
        TableDihedralForceComputeGPU.cc
        TableDihedralForceGPU.cu
        TriangleAreaConservationMeshForceComputeGPU.cc
        TriangleAreaConservationMeshForceComputeGPU.cu
        TwoStepBDGPU.cc
        TwoStepBDGPU.cu
        TwoStepConstantPressureGPU.cc
        TwoStepConstantPressureGPU.cu
        TwoStepConstantVolumeGPU.cc
        TwoStepConstantVolumeGPU.cu
        TwoStepLangevinGPU.cc
        TwoStepLangevinGPU.cu
        TwoStepNVEGPU.cu
        TwoStepRATTLELangevinGPU.cu
        TwoStepRATTLENVEGPU.cu
        VolumeConservationMeshForceComputeGPU.cc
        VolumeConservationMeshForceComputeGPU.cu
        ${DFFT_CU_SOURCES}
        )
    list(APPEND _md_headers
        ActiveForceComputeGPU.h
        ActiveForceConstraintComputeGPU.h
        ActiveForceConstraintComputeGPU.cuh
        AnisoPotentialPairGPU.cuh
        AnisoPotentialPairGPU.h
        AreaConservationMeshForceComputeGPU.h
        AreaConservationMeshForceComputeGPU.cuh
        BendingRigidityMeshForceComputeGPU.h
        BondTablePotentialGPU.h
        BoxDeformerGPU.cuh
        CommunicatorGridGPU.h
        ComputeThermoGPU.cuh
        ComputeThermoGPU.h
        ComputeThermoHMAGPU.cuh
        ComputeThermoHMAGPU.h
        ConstantForceComputeGPU.h
        ConstantForceComputeGPU.cuh
        CosineSqAngleForceComputeGPU.h
        CosineSqAngleForceGPU.cuh
        FIREEnergyMinimizerGPU.h
        FIREEnergyMinimizerGPU.cuh
        ForceCompositeGPU.h
        ForceCompositeGPU.cuh
        ForceDistanceConstraintGPU.h
        ForceDistanceConstraintGPU.cuh
        HarmonicAngleForceComputeGPU.h
        HarmonicAngleForceGPU.cuh
        HarmonicDihedralForceComputeGPU.h
        HarmonicDihedralForceGPU.cuh
        HarmonicImproperForceComputeGPU.h
        HarmonicImproperForceGPU.cuh
        HelfrichMeshForceComputeGPU.h
        HelfrichMeshForceComputeGPU.cuh
        LeesEdwardsBoxDeformerGPU.cuh
        MolecularForceCompute.cuh
        MuellerPlatheFlowGPU.h
        MuellerPlatheFlowGPU.cuh
        NeighborListGPU.h
        NeighborListGPU.cuh
        NeighborListGPUBinned.h
        NeighborListGPUBinned.cuh
        NeighborListGPUStencil.h
        NeighborListGPUStencil.cuh
        NeighborListGPUTree.h
        NeighborListGPUTree.cuh
        OPLSDihedralForceComputeGPU.h
        OPLSDihedralForceGPU.cuh
        PotentialBondGPU.h
        PotentialBondGPU.cuh
        PotentialExternalGPU.h
        PotentialExternalGPU.cuh
        PotentialPairGPU.h
        PotentialPairGPU.cuh
        PotentialPairDPDThermoGPU.h
        PotentialPairDPDThermoGPU.cuh
        PotentialSpecialPairGPU.h
        PotentialTersoffGPU.h
        PeriodicImproperForceComputeGPU.h
        PeriodicImproperForceGPU.cuh
        PPPMForceComputeGPU.h
        PPPMForceComputeGPU.cuh
        FrictionPairGPU.cuh
        FrictionPairGPU.h
        TableAngleForceComputeGPU.h
        TableAngleForceGPU.cuh
        TableDihedralForceComputeGPU.h
        TriangleAreaConservationMeshForceComputeGPU.h
        TriangleAreaConservationMeshForceComputeGPU.cuh
        TwoStepBDGPU.h
        TwoStepConstantPressureGPU.h
        TwoStepConstantPressureGPU.cuh
        TwoStepConstantVolumeGPU.h
        TwoStepConstantVolumeGPU.cuh
        TwoStepLangevinGPU.h
        TwoStepLangevinGPU.cuh
        TwoStepNVEGPU.cuh
        TwoStepRATTLEBDGPU.h
        TwoStepRATTLEBDGPU.cuh
        TwoStepRATTLELangevinGPU.h
        TwoStepRATTLELangevinGPU.cuh
        TwoStepRATTLENVEGPU.h
        TwoStepRATTLENVEGPU.cuh
        VolumeConservationMeshForceComputeGPU.h
        VolumeConservationMeshForceComputeGPU.cuh
        )
endif()

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
    list(APPEND _md_sources export_ActiveForceConstraintCompute${_manifold}.cc)

    if (ENABLE_GPU)
        configure_file(export_ActiveForceConstraintComputeGPU.cc.inc
                       export_ActiveForceConstraintCompute${_manifold}GPU.cc
                       @ONLY)
        configure_file(ActiveForceConstraintComputeGPU.cu.inc
                       ActiveForceConstraintCompute${_manifold}GPU.cu
                       @ONLY)
        configure_file(TwoStepRATTLEGPU.cu.inc
                       TwoStepRATTLE${_manifold}GPU.cu
                       @ONLY)
        list(APPEND _md_sources
            export_ActiveForceConstraintCompute${_manifold}GPU.cc
            ActiveForceConstraintCompute${_manifold}GPU.cu
            TwoStepRATTLE${_manifold}GPU.cu
            )
    endif()

    foreach (_method ${_rattle_methods})
        configure_file(export_TwoStepRATTLE.cc.inc
                    export_TwoStepRATTLE${_method}${_manifold}.cc
                    @ONLY)
        list(APPEND _md_sources export_TwoStepRATTLE${_method}${_manifold}.cc)

        if (ENABLE_GPU)
            configure_file(export_TwoStepRATTLEGPU.cc.inc
                        export_TwoStepRATTLE${_method}${_manifold}GPU.cc
                        @ONLY)
            list(APPEND _md_sources export_TwoStepRATTLE${_method}${_manifold}GPU.cc)
        endif()
    endforeach()

endforeach()

# generate pybind11 export cc files
set(_bonds Harmonic FENE Tether)

foreach(_bond ${_bonds})
    configure_file(export_PotentialBond.cc.inc
                   export_PotentialBond${_bond}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialBond${_bond}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialBondGPU.cc.inc
                       export_PotentialBond${_bond}GPU.cc
                       @ONLY)
        configure_file(PotentialBondGPUKernel.cu.inc
                       PotentialBond${_bond}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_PotentialBond${_bond}GPU.cc
            PotentialBond${_bond}GPUKernel.cu
            )
    endif()
endforeach()

foreach(_bond ${_bonds})
    configure_file(export_PotentialMeshBond.cc.inc
                   export_PotentialMeshBond${_bond}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialMeshBond${_bond}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialMeshBondGPU.cc.inc
                       export_PotentialMeshBond${_bond}GPU.cc
                       @ONLY)
        configure_file(PotentialMeshBondGPUKernel.cu.inc
                       PotentialMeshBond${_bond}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_PotentialMeshBond${_bond}GPU.cc
            PotentialMeshBond${_bond}GPUKernel.cu
            )
    endif()
endforeach()

set(_pairs LJ Coulomb)

foreach(_pair ${_pairs})
    configure_file(export_PotentialSpecialPair.cc.inc
                   export_PotentialSpecialPair${_pair}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialSpecialPair${_pair}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialSpecialPairGPU.cc.inc
                       export_PotentialSpecialPair${_pair}GPU.cc
                       @ONLY)
        configure_file(PotentialSpecialPairGPUKernel.cu.inc
                       PotentialSpecialPair${_pair}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
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
    list(APPEND _md_sources export_PotentialTersoff${_evaluator}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialTersoffGPU.cc.inc
                       export_PotentialTersoff${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialTersoffGPUKernel.cu.inc
                       PotentialTersoff${_evaluator}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_PotentialTersoff${_evaluator}GPU.cc
            PotentialTersoff${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

set(_external_evaluators Periodic ElectricField MagneticField)

foreach(_evaluator ${_external_evaluators})
    configure_file(export_PotentialExternal.cc.inc
                   export_PotentialExternal${_evaluator}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialExternal${_evaluator}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialExternalGPU.cc.inc
                       export_PotentialExternal${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialExternalGPUKernel.cu.inc
                       PotentialExternal${_evaluator}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
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
    list(APPEND _md_sources export_PotentialExternalWall${_evaluator}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialExternalWallGPU.cc.inc
                       export_PotentialExternalWall${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialExternalWallGPUKernel.cu.inc
                       PotentialExternalWall${_evaluator}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_PotentialExternalWall${_evaluator}GPU.cc
            PotentialExternalWall${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

set(_dpdthermo_evaluators DPD LJ)

foreach(_evaluator ${_dpdthermo_evaluators})
    configure_file(export_PotentialPairDPDThermo.cc.inc
                   export_PotentialPairDPDThermo${_evaluator}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialPairDPDThermo${_evaluator}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialPairDPDThermoGPU.cc.inc
                       export_PotentialPairDPDThermo${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialPairDPDThermoGPUKernel.cu.inc
                       PotentialPairDPDThermo${_evaluator}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_PotentialPairDPDThermo${_evaluator}GPU.cc
            PotentialPairDPDThermo${_evaluator}GPUKernel.cu
            )
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
                     ExpandedGaussian
                     WangFrenkel
                     Zetterling)


foreach(_evaluator ${_pair_evaluators})
    set(_evaluator_cpp ${_evaluator})
    if (_evaluator STREQUAL "ConservativeDPD")
        set(_evaluator_cpp "DPDThermoDPD")
    endif()

    configure_file(export_PotentialPair.cc.inc
                   export_PotentialPair${_evaluator}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialPair${_evaluator}.cc)

    if (ENABLE_GPU)
        configure_file(export_PotentialPairGPU.cc.inc
                       export_PotentialPair${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialPairGPUKernel.cu.inc
                       PotentialPair${_evaluator}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_PotentialPair${_evaluator}GPU.cc
            PotentialPair${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

set(_patchy_evaluators ExpandedGaussian
                       ExpandedLJ
                       ExpandedMie
                       Gauss
                       LJ
                       Mie
                       Yukawa
                       Table)

foreach(_evaluator ${_patchy_evaluators})
    configure_file(export_AnisoPotentialPairPatchy.cc.inc
                   export_AnisoPotentialPairPatchy${_evaluator}.cc
                   @ONLY)
    list(APPEND _md_sources export_AnisoPotentialPairPatchy${_evaluator}.cc)
    if(ENABLE_GPU)
        configure_file(export_AnisoPotentialPairPatchyGPU.cc.inc
                       export_AnisoPotentialPairPatchy${_evaluator}GPU.cc
                       @ONLY)
        configure_file(AnisoPotentialPairPatchyGPUKernel.cu.inc
                       AnisoPotentialPairPatchy${_evaluator}GPUKernel.cu
                       @ONLY)
        list(APPEND _md_sources
            export_AnisoPotentialPairPatchy${_evaluator}GPU.cc
            AnisoPotentialPairPatchy${_evaluator}GPUKernel.cu
            )
    endif()
endforeach()

set(_alchemical_pair_evaluators LJGauss)

foreach(_evaluator ${_alchemical_pair_evaluators})
    configure_file(export_PotentialPairAlchemical.cc.inc
                   export_PotentialPairAlchemical${_evaluator}.cc
                   @ONLY)
    list(APPEND _md_sources export_PotentialPairAlchemical${_evaluator}.cc)
endforeach()

hoomd_add_module(_md SHARED ${_md_sources} ${_md_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_md ALIAS _md)

if(APPLE)
set_target_properties(_md PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_md PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_md PUBLIC _hoomd)
if (ENABLE_GPU)
    if (TARGET CUDA::cusolver)
        target_link_libraries(_md PUBLIC CUDA::cusolver CUDA::cublas CUDA::cusparse)
        target_compile_definitions(_md PUBLIC CUSOLVER_AVAILABLE)
    endif()
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
          deform.py
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
