// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveRotationalDiffusionUpdater.h"
#include "AllBondPotentials.h"
#include "AllExternalPotentials.h"
#include "AllPairPotentials.h"
#include "AllSpecialPairPotentials.h"
#include "AllTripletPotentials.h"
#include "BondTablePotential.h"
#include "ComputeThermo.h"
#include "ComputeThermoHMA.h"
#include "CosineSqAngleForceCompute.h"
#include "CustomForceCompute.h"
#include "EvaluatorRevCross.h"
#include "EvaluatorSquareDensity.h"
#include "EvaluatorTersoff.h"
#include "FIREEnergyMinimizer.h"
#include "ForceComposite.h"
#include "ForceDistanceConstraint.h"
#include "HarmonicAngleForceCompute.h"
#include "HarmonicDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"
#include "IntegrationMethodTwoStep.h"
#include "IntegratorTwoStep.h"
#include "ManifoldDiamond.h"
#include "ManifoldEllipsoid.h"
#include "ManifoldGyroid.h"
#include "ManifoldPrimitive.h"
#include "ManifoldSphere.h"
#include "ManifoldXYPlane.h"
#include "ManifoldZCylinder.h"
#include "MolecularForceCompute.h"
#include "MuellerPlatheFlow.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "NeighborListStencil.h"
#include "NeighborListTree.h"
#include "OPLSDihedralForceCompute.h"
#include "PPPMForceCompute.h"
#include "PotentialBond.h"
#include "PotentialExternal.h"
#include "PotentialPair.h"
#include "PotentialPairDPDThermo.h"
#include "PotentialTersoff.h"
#include "TableAngleForceCompute.h"
#include "TableDihedralForceCompute.h"
#include "TwoStepBD.h"
#include "TwoStepBerendsen.h"
#include "TwoStepLangevin.h"
#include "TwoStepLangevinBase.h"
#include "TwoStepNPTMTK.h"
#include "TwoStepNVE.h"
#include "TwoStepNVTMTK.h"
#include "TwoStepRATTLEBD.h"
#include "TwoStepRATTLELangevin.h"
#include "TwoStepRATTLENVE.h"
#include "WallData.h"
#include "ZeroMomentumUpdater.h"

// include GPU classes
#ifdef ENABLE_HIP
#include "BondTablePotentialGPU.h"
#include "ComputeThermoGPU.h"
#include "ComputeThermoHMAGPU.h"
#include "CosineSqAngleForceComputeGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#include "ForceCompositeGPU.h"
#include "ForceDistanceConstraintGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#include "HarmonicDihedralForceComputeGPU.h"
#include "HarmonicImproperForceComputeGPU.h"
#include "MuellerPlatheFlowGPU.h"
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
#include "NeighborListGPUStencil.h"
#include "NeighborListGPUTree.h"
#include "OPLSDihedralForceComputeGPU.h"
#include "PPPMForceComputeGPU.h"
#include "PotentialBondGPU.h"
#include "PotentialExternalGPU.h"
#include "PotentialPairDPDThermoGPU.h"
#include "PotentialPairGPU.h"
#include "PotentialTersoffGPU.h"
#include "TableAngleForceComputeGPU.h"
#include "TableDihedralForceComputeGPU.h"
#include "TwoStepBDGPU.h"
#include "TwoStepBerendsenGPU.h"
#include "TwoStepLangevinGPU.h"
#include "TwoStepNPTMTKGPU.h"
#include "TwoStepNVEGPU.h"
#include "TwoStepNVTMTKGPU.h"
#include "TwoStepRATTLEBDGPU.h"
#include "TwoStepRATTLELangevinGPU.h"
#include "TwoStepRATTLENVEGPU.h"
#endif

#include <pybind11/pybind11.h>


namespace hoomd { namespace md { namespace detail {

void export_ActiveForceCompute(pybind11::module &m);
void export_ActiveForceConstraintComputeCylinder(pybind11::module &m);
void export_ActiveForceConstraintComputeDiamond(pybind11::module &m);
void export_ActiveForceConstraintComputeEllipsoid(pybind11::module &m);
void export_ActiveForceConstraintComputeGyroid(pybind11::module &m);
void export_ActiveForceConstraintComputePlane(pybind11::module &m);
void export_ActiveForceConstraintComputePrimitive(pybind11::module &m);
void export_ActiveForceConstraintComputeSphere(pybind11::module &m);
void export_ActiveRotationalDiffusionUpdater(pybind11::module &m);
void export_AnisoPotentialPairALJ2D(pybind11::module &m);
void export_AnisoPotentialPairALJ3D(pybind11::module &m);
void export_AnisoPotentialPairDipole(pybind11::module &m);
void export_AnisoPotentialPairGB(pybind11::module &m);

#ifdef ENABLE_HIP

void export_ActiveForceConstraintComputeCylinderGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeDiamondGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeEllipsoidGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeGyroidGPU(pybind11::module &m);
void export_ActiveForceConstraintComputePlaneGPU(pybind11::module &m);
void export_ActiveForceConstraintComputePrimitiveGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeSphereGPU(pybind11::module &m);
void export_ActiveForceComputeGPU(pybind11::module &m);
void export_AnisoPotentialPairALJ2DGPU(pybind11::module &m);
void export_AnisoPotentialPairALJ3DGPU(pybind11::module &m);
void export_AnisoPotentialPairDipoleGPU(pybind11::module &m);
void export_AnisoPotentialPairGBGPU(pybind11::module &m);
#endif
} } }

using namespace hoomd;
using namespace hoomd::md;
using namespace hoomd::md::detail;

//! Export setParamsPython and getParams as a different name
// Electric field only has one parameter, so we can get its parameter from
// python with by a name other than getParams and setParams
template<>
void hoomd::md::detail::export_PotentialExternal<PotentialExternalElectricField>(
    pybind11::module& m,
    const std::string& name)
    {
    pybind11::class_<PotentialExternalElectricField,
                     ForceCompute,
                     std::shared_ptr<PotentialExternalElectricField>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setE", &PotentialExternalElectricField::setParamsPython)
        .def("getE", &PotentialExternalElectricField::getParams);
    }

// Simplify the exporting of wall potential subclasses
template<class EvaluatorPairType>
void export_WallPotential(pybind11::module& m, const std::string& name)
    {
    export_PotentialExternal<PotentialExternal<EvaluatorWalls<EvaluatorPairType>>>(m, name);
    }

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
PYBIND11_MODULE(_md, m)
    {
    export_ActiveForceCompute(m);
    export_ActiveForceConstraintComputeCylinder(m);
    export_ActiveForceConstraintComputeDiamond(m);
    export_ActiveForceConstraintComputeEllipsoid(m);
    export_ActiveForceConstraintComputeGyroid(m);
    export_ActiveForceConstraintComputePlane(m);
    export_ActiveForceConstraintComputePrimitive(m);
    export_ActiveForceConstraintComputeSphere(m);
    export_ActiveRotationalDiffusionUpdater(m);
    export_ComputeThermo(m);
    export_ComputeThermoHMA(m);
    export_HarmonicAngleForceCompute(m);
    export_CosineSqAngleForceCompute(m);
    export_TableAngleForceCompute(m);
    export_HarmonicDihedralForceCompute(m);
    export_OPLSDihedralForceCompute(m);
    export_TableDihedralForceCompute(m);
    export_HarmonicImproperForceCompute(m);
    export_BondTablePotential(m);
    export_PotentialPair<PotentialPairBuckingham>(m, "PotentialPairBuckingham");
    export_PotentialPair<PotentialPairLJ>(m, "PotentialPairLJ");
    export_PotentialPair<PotentialPairLJ1208>(m, "PotentialPairLJ1208");
    export_PotentialPair<PotentialPairLJ0804>(m, "PotentialPairLJ0804");
    export_PotentialPair<PotentialPairGauss>(m, "PotentialPairGauss");
    export_PotentialPair<PotentialPairExpandedLJ>(m, "PotentialPairExpandedLJ");
    export_PotentialPair<PotentialPairExpandedMie>(m, "PotentialPairExpandedMie");
    export_PotentialPair<PotentialPairYukawa>(m, "PotentialPairYukawa");
    export_PotentialPair<PotentialPairEwald>(m, "PotentialPairEwald");
    export_PotentialPair<PotentialPairMorse>(m, "PotentialPairMorse");
    export_PotentialPair<PotentialPairDPD>(m, "PotentialPairDPD");
    export_PotentialPair<PotentialPairMoliere>(m, "PotentialPairMoliere");
    export_PotentialPair<PotentialPairZBL>(m, "PotentialPairZBL");
    export_PotentialTersoff<PotentialTripletTersoff>(m, "PotentialTersoff");
    export_PotentialTersoff<PotentialTripletSquareDensity>(m, "PotentialSquareDensity");
    export_PotentialTersoff<PotentialTripletRevCross>(m, "PotentialRevCross");
    export_PotentialPair<PotentialPairMie>(m, "PotentialPairMie");
    export_PotentialPair<PotentialPairReactionField>(m, "PotentialPairReactionField");
    export_PotentialPair<PotentialPairDLVO>(m, "PotentialPairDLVO");
    export_PotentialPair<PotentialPairFourier>(m, "PotentialPairFourier");
    export_PotentialPair<PotentialPairOPP>(m, "PotentialPairOPP");
    export_PotentialPair<PotentialPairTWF>(m, "PotentialPairTWF");
    export_AnisoPotentialPairALJ2D(m);
    export_AnisoPotentialPairALJ3D(m);
    export_AnisoPotentialPairDipole(m);
    export_AnisoPotentialPairGB(m);
    export_PotentialPair<PotentialPairForceShiftedLJ>(m, "PotentialPairForceShiftedLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDThermoDPD, PotentialPairDPD>(
        m,
        "PotentialPairDPDThermoDPD");
    export_PotentialPair<PotentialPairDPDLJ>(m, "PotentialPairDPDLJ");
    export_PotentialPair<PotentialPairTable>(m, "PotentialPairTable");
    export_PotentialPairDPDThermo<PotentialPairDPDLJThermoDPD, PotentialPairDPDLJ>(
        m,
        "PotentialPairDPDLJThermoDPD");
    export_PotentialBond<PotentialBondHarmonic>(m, "PotentialBondHarmonic");
    export_PotentialBond<PotentialBondFENE>(m, "PotentialBondFENE");
    export_PotentialBond<PotentialBondTether>(m, "PotentialBondTether");
    export_PotentialSpecialPair<PotentialSpecialPairLJ>(m, "PotentialSpecialPairLJ");
    export_PotentialSpecialPair<PotentialSpecialPairCoulomb>(m, "PotentialSpecialPairCoulomb");
    export_CustomForceCompute(m);
    export_NeighborList(m);
    export_NeighborListBinned(m);
    export_NeighborListStencil(m);
    export_NeighborListTree(m);
    export_MolecularForceCompute(m);
    export_ForceDistanceConstraint(m);
    export_ForceComposite(m);
    export_PPPMForceCompute(m);
    export_PotentialExternal<PotentialExternalPeriodic>(m, "PotentialExternalPeriodic");
    export_PotentialExternal<PotentialExternalElectricField>(m, "PotentialExternalElectricField");
    export_wall_data(m);
    export_wall_field(m);
    export_WallPotential<EvaluatorPairLJ>(m, "WallsPotentialLJ");
    export_WallPotential<EvaluatorPairYukawa>(m, "WallsPotentialYukawa");
    export_WallPotential<EvaluatorPairForceShiftedLJ>(m, "WallsPotentialForceShiftedLJ");
    export_WallPotential<EvaluatorPairMie>(m, "WallsPotentialMie");
    export_WallPotential<EvaluatorPairGauss>(m, "WallsPotentialGauss");
    export_WallPotential<EvaluatorPairMorse>(m, "WallsPotentialMorse");

#ifdef ENABLE_HIP
    export_NeighborListGPU(m);
    export_NeighborListGPUBinned(m);
    export_NeighborListGPUStencil(m);
    export_NeighborListGPUTree(m);
    export_ForceCompositeGPU(m);
    export_PotentialPairGPU<PotentialPairBuckinghamGPU, PotentialPairBuckingham>(
        m,
        "PotentialPairBuckinghamGPU");
    export_PotentialPairGPU<PotentialPairLJGPU, PotentialPairLJ>(m, "PotentialPairLJGPU");
    export_PotentialPairGPU<PotentialPairLJ1208GPU, PotentialPairLJ1208>(m,
                                                                         "PotentialPairLJ1208GPU");
    export_PotentialPairGPU<PotentialPairLJ0804GPU, PotentialPairLJ0804>(m,
                                                                         "PotentialPairLJ0804GPU");
    export_PotentialPairGPU<PotentialPairGaussGPU, PotentialPairGauss>(m, "PotentialPairGaussGPU");
    export_PotentialPairGPU<PotentialPairExpandedLJGPU, PotentialPairExpandedLJ>(
        m,
        "PotentialPairExpandedLJGPU");
    export_PotentialPairGPU<PotentialPairYukawaGPU, PotentialPairYukawa>(m,
                                                                         "PotentialPairYukawaGPU");
    export_PotentialPairGPU<PotentialPairReactionFieldGPU, PotentialPairReactionField>(
        m,
        "PotentialPairReactionFieldGPU");
    export_PotentialPairGPU<PotentialPairDLVOGPU, PotentialPairDLVO>(m, "PotentialPairDLVOGPU");
    export_PotentialPairGPU<PotentialPairFourierGPU, PotentialPairFourier>(
        m,
        "PotentialPairFourierGPU");
    export_PotentialPairGPU<PotentialPairEwaldGPU, PotentialPairEwald>(m, "PotentialPairEwaldGPU");
    export_PotentialPairGPU<PotentialPairMorseGPU, PotentialPairMorse>(m, "PotentialPairMorseGPU");
    export_PotentialPairGPU<PotentialPairDPDGPU, PotentialPairDPD>(m, "PotentialPairDPDGPU");
    export_PotentialPairGPU<PotentialPairMoliereGPU, PotentialPairMoliere>(
        m,
        "PotentialPairMoliereGPU");
    export_PotentialPairGPU<PotentialPairZBLGPU, PotentialPairZBL>(m, "PotentialPairZBLGPU");
    export_PotentialTersoffGPU<PotentialTripletTersoffGPU, PotentialTripletTersoff>(
        m,
        "PotentialTersoffGPU");
    export_PotentialTersoffGPU<PotentialTripletSquareDensityGPU, PotentialTripletSquareDensity>(
        m,
        "PotentialSquareDensityGPU");
    export_PotentialTersoffGPU<PotentialTripletRevCrossGPU, PotentialTripletRevCross>(
        m,
        "PotentialRevCrossGPU");
    export_PotentialPairGPU<PotentialPairForceShiftedLJGPU, PotentialPairForceShiftedLJ>(
        m,
        "PotentialPairForceShiftedLJGPU");
    export_PotentialPairGPU<PotentialPairMieGPU, PotentialPairMie>(m, "PotentialPairMieGPU");
    export_PotentialPairGPU<PotentialPairExpandedMieGPU, PotentialPairExpandedMie>(
        m,
        "PotentialPairExpandedMieGPU");
    export_PotentialPairGPU<PotentialPairOPPGPU, PotentialPairOPP>(m, "PotentialPairOPPGPU");
    export_PotentialPairGPU<PotentialPairTWFGPU, PotentialPairTWF>(m, "PotentialPairTWFGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDThermoDPDGPU, PotentialPairDPDThermoDPD>(
        m,
        "PotentialPairDPDThermoDPDGPU");
    export_PotentialPairGPU<PotentialPairDPDLJGPU, PotentialPairDPDLJ>(m, "PotentialPairDPDLJGPU");
    export_PotentialPairGPU<PotentialPairTableGPU, PotentialPairTable>(m, "PotentialPairTableGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDLJThermoDPDGPU, PotentialPairDPDLJThermoDPD>(
        m,
        "PotentialPairDPDLJThermoDPDGPU");
    export_AnisoPotentialPairALJ2DGPU(m);
    export_AnisoPotentialPairALJ3DGPU(m);
    export_AnisoPotentialPairDipoleGPU(m);
    export_AnisoPotentialPairGBGPU(m);
    export_PotentialBondGPU<PotentialBondHarmonicGPU, PotentialBondHarmonic>(
        m,
        "PotentialBondHarmonicGPU");
    export_PotentialBondGPU<PotentialBondFENEGPU, PotentialBondFENE>(m, "PotentialBondFENEGPU");
    export_PotentialBondGPU<PotentialBondTetherGPU, PotentialBondTether>(m,
                                                                         "PotentialBondTetherGPU");
    export_PotentialSpecialPairGPU<PotentialSpecialPairLJGPU, PotentialSpecialPairLJ>(
        m,
        "PotentialSpecialPairLJGPU");
    export_PotentialSpecialPairGPU<PotentialSpecialPairCoulombGPU, PotentialSpecialPairCoulomb>(
        m,
        "PotentialSpecialPairCoulombGPU");
    export_BondTablePotentialGPU(m);
    export_HarmonicAngleForceComputeGPU(m);
    export_CosineSqAngleForceComputeGPU(m);
    export_TableAngleForceComputeGPU(m);
    export_HarmonicDihedralForceComputeGPU(m);
    export_OPLSDihedralForceComputeGPU(m);
    export_TableDihedralForceComputeGPU(m);
    export_HarmonicImproperForceComputeGPU(m);
    export_ForceDistanceConstraintGPU(m);
    export_ComputeThermoGPU(m);
    export_ComputeThermoHMAGPU(m);
    export_PPPMForceComputeGPU(m);
    export_ActiveForceComputeGPU(m);
    export_ActiveForceConstraintComputeCylinderGPU(
        m);
    export_ActiveForceConstraintComputeDiamondGPU(
        m);
    export_ActiveForceConstraintComputeEllipsoidGPU(
        m);
    export_ActiveForceConstraintComputeGyroidGPU(m);
    export_ActiveForceConstraintComputePlaneGPU(m);
    export_ActiveForceConstraintComputePrimitiveGPU(m);
    export_ActiveForceConstraintComputeSphereGPU(m);
    export_PotentialExternalGPU<PotentialExternalPeriodicGPU, PotentialExternalPeriodic>(
        m,
        "PotentialExternalPeriodicGPU");
    export_PotentialExternalGPU<PotentialExternalElectricFieldGPU, PotentialExternalElectricField>(
        m,
        "PotentialExternalElectricFieldGPU");

    export_PotentialExternalGPU<WallsPotentialLJGPU, WallsPotentialLJ>(m, "WallsPotentialLJGPU");
    export_PotentialExternalGPU<WallsPotentialYukawaGPU, WallsPotentialYukawa>(
        m,
        "WallsPotentialYukawaGPU");
    export_PotentialExternalGPU<WallsPotentialForceShiftedLJGPU, WallsPotentialForceShiftedLJ>(
        m,
        "WallsPotentialForceShiftedLJGPU");
    export_PotentialExternalGPU<WallsPotentialMieGPU, WallsPotentialMie>(m, "WallsPotentialMieGPU");
    export_PotentialExternalGPU<WallsPotentialGaussGPU, WallsPotentialGauss>(
        m,
        "WallsPotentialGaussGPU");
    export_PotentialExternalGPU<WallsPotentialMorseGPU, WallsPotentialMorse>(
        m,
        "WallsPotentialMorseGPU");
#endif

    // updaters
    export_IntegratorTwoStep(m);
    export_IntegrationMethodTwoStep(m);
    export_ZeroMomentumUpdater(m);
    export_TwoStepNVE(m);
    export_TwoStepNVTMTK(m);
    export_TwoStepLangevinBase(m);
    export_TwoStepLangevin(m);
    export_TwoStepBD(m);
    export_TwoStepNPTMTK(m);
    export_Berendsen(m);
    export_FIREEnergyMinimizer(m);
    export_MuellerPlatheFlow(m);

    // RATTLE
    export_TwoStepRATTLEBD<ManifoldZCylinder>(m, "TwoStepRATTLEBDCylinder");
    export_TwoStepRATTLEBD<ManifoldDiamond>(m, "TwoStepRATTLEBDDiamond");
    export_TwoStepRATTLEBD<ManifoldEllipsoid>(m, "TwoStepRATTLEBDEllipsoid");
    export_TwoStepRATTLEBD<ManifoldGyroid>(m, "TwoStepRATTLEBDGyroid");
    export_TwoStepRATTLEBD<ManifoldXYPlane>(m, "TwoStepRATTLEBDPlane");
    export_TwoStepRATTLEBD<ManifoldPrimitive>(m, "TwoStepRATTLEBDPrimitive");
    export_TwoStepRATTLEBD<ManifoldSphere>(m, "TwoStepRATTLEBDSphere");

    export_TwoStepRATTLELangevin<ManifoldZCylinder>(m, "TwoStepRATTLELangevinCylinder");
    export_TwoStepRATTLELangevin<ManifoldDiamond>(m, "TwoStepRATTLELangevinDiamond");
    export_TwoStepRATTLELangevin<ManifoldEllipsoid>(m, "TwoStepRATTLELangevinEllipsoid");
    export_TwoStepRATTLELangevin<ManifoldGyroid>(m, "TwoStepRATTLELangevinGyroid");
    export_TwoStepRATTLELangevin<ManifoldXYPlane>(m, "TwoStepRATTLELangevinPlane");
    export_TwoStepRATTLELangevin<ManifoldPrimitive>(m, "TwoStepRATTLELangevinPrimitive");
    export_TwoStepRATTLELangevin<ManifoldSphere>(m, "TwoStepRATTLELangevinSphere");

    export_TwoStepRATTLENVE<ManifoldZCylinder>(m, "TwoStepRATTLENVECylinder");
    export_TwoStepRATTLENVE<ManifoldDiamond>(m, "TwoStepRATTLENVEDiamond");
    export_TwoStepRATTLENVE<ManifoldEllipsoid>(m, "TwoStepRATTLENVEEllipsoid");
    export_TwoStepRATTLENVE<ManifoldGyroid>(m, "TwoStepRATTLENVEGyroid");
    export_TwoStepRATTLENVE<ManifoldXYPlane>(m, "TwoStepRATTLENVEPlane");
    export_TwoStepRATTLENVE<ManifoldPrimitive>(m, "TwoStepRATTLENVEPrimitive");
    export_TwoStepRATTLENVE<ManifoldSphere>(m, "TwoStepRATTLENVESphere");

#ifdef ENABLE_HIP
    export_TwoStepNVEGPU(m);
    export_TwoStepNVTMTKGPU(m);
    export_TwoStepLangevinGPU(m);
    export_TwoStepBDGPU(m);
    export_TwoStepNPTMTKGPU(m);
    export_BerendsenGPU(m);
    export_FIREEnergyMinimizerGPU(m);
    export_MuellerPlatheFlowGPU(m);

    export_TwoStepRATTLEBDGPU<ManifoldZCylinder>(m, "TwoStepRATTLEBDCylinderGPU");
    export_TwoStepRATTLEBDGPU<ManifoldDiamond>(m, "TwoStepRATTLEBDDiamondGPU");
    export_TwoStepRATTLEBDGPU<ManifoldEllipsoid>(m, "TwoStepRATTLEBDEllipsoidGPU");
    export_TwoStepRATTLEBDGPU<ManifoldGyroid>(m, "TwoStepRATTLEBDGyroidGPU");
    export_TwoStepRATTLEBDGPU<ManifoldXYPlane>(m, "TwoStepRATTLEBDPlaneGPU");
    export_TwoStepRATTLEBDGPU<ManifoldPrimitive>(m, "TwoStepRATTLEBDPrimitiveGPU");
    export_TwoStepRATTLEBDGPU<ManifoldSphere>(m, "TwoStepRATTLEBDSphereGPU");

    export_TwoStepRATTLELangevinGPU<ManifoldZCylinder>(m, "TwoStepRATTLELangevinCylinderGPU");
    export_TwoStepRATTLELangevinGPU<ManifoldDiamond>(m, "TwoStepRATTLELangevinDiamondGPU");
    export_TwoStepRATTLELangevinGPU<ManifoldEllipsoid>(m, "TwoStepRATTLELangevinEllipsoidGPU");
    export_TwoStepRATTLELangevinGPU<ManifoldGyroid>(m, "TwoStepRATTLELangevinGyroidGPU");
    export_TwoStepRATTLELangevinGPU<ManifoldXYPlane>(m, "TwoStepRATTLELangevinPlaneGPU");
    export_TwoStepRATTLELangevinGPU<ManifoldPrimitive>(m, "TwoStepRATTLELangevinPrimitiveGPU");
    export_TwoStepRATTLELangevinGPU<ManifoldSphere>(m, "TwoStepRATTLELangevinSphereGPU");

    export_TwoStepRATTLENVEGPU<ManifoldZCylinder>(m, "TwoStepRATTLENVECylinderGPU");
    export_TwoStepRATTLENVEGPU<ManifoldDiamond>(m, "TwoStepRATTLENVEDiamondGPU");
    export_TwoStepRATTLENVEGPU<ManifoldEllipsoid>(m, "TwoStepRATTLENVEEllipsoidGPU");
    export_TwoStepRATTLENVEGPU<ManifoldGyroid>(m, "TwoStepRATTLENVEGyroidGPU");
    export_TwoStepRATTLENVEGPU<ManifoldXYPlane>(m, "TwoStepRATTLENVEPlaneGPU");
    export_TwoStepRATTLENVEGPU<ManifoldPrimitive>(m, "TwoStepRATTLENVEPrimitiveGPU");
    export_TwoStepRATTLENVEGPU<ManifoldSphere>(m, "TwoStepRATTLENVESphereGPU");
#endif

    // manifolds
    export_ManifoldZCylinder(m);
    export_ManifoldDiamond(m);
    export_ManifoldEllipsoid(m);
    export_ManifoldGyroid(m);
    export_ManifoldXYPlane(m);
    export_ManifoldPrimitive(m);
    export_ManifoldSphere(m);
    }
