// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AllExternalPotentials.h"
#include "AllPairPotentials.h"
#include "AllSpecialPairPotentials.h"
#include "AllTripletPotentials.h"
#include "EvaluatorRevCross.h"
#include "EvaluatorSquareDensity.h"
#include "EvaluatorTersoff.h"
#include "PotentialExternal.h"
#include "PotentialPair.h"
#include "PotentialPairDPDThermo.h"
#include "PotentialTersoff.h"

// include GPU classes
#ifdef ENABLE_HIP
#include "PotentialExternalGPU.h"
#include "PotentialPairDPDThermoGPU.h"
#include "PotentialPairGPU.h"
#include "PotentialTersoffGPU.h"
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
void export_ComputeThermo(pybind11::module& m);
void export_ComputeThermoHMA(pybind11::module& m);
void export_HarmonicAngleForceCompute(pybind11::module& m);
void export_CosineSqAngleForceCompute(pybind11::module& m);
void export_TableAngleForceCompute(pybind11::module& m);
void export_HarmonicDihedralForceCompute(pybind11::module& m);
void export_OPLSDihedralForceCompute(pybind11::module& m);
void export_TableDihedralForceCompute(pybind11::module& m);
void export_HarmonicImproperForceCompute(pybind11::module& m);
void export_BondTablePotential(pybind11::module& m);
void export_CustomForceCompute(pybind11::module& m);
void export_NeighborList(pybind11::module& m);
void export_NeighborListBinned(pybind11::module& m);
void export_NeighborListStencil(pybind11::module& m);
void export_NeighborListTree(pybind11::module& m);
void export_MolecularForceCompute(pybind11::module& m);
void export_ForceDistanceConstraint(pybind11::module& m);
void export_ForceComposite(pybind11::module& m);
void export_PPPMForceCompute(pybind11::module& m);
void export_wall_data(pybind11::module& m);
void export_wall_field(pybind11::module& m);

void export_AnisoPotentialPairALJ2D(pybind11::module &m);
void export_AnisoPotentialPairALJ3D(pybind11::module &m);
void export_AnisoPotentialPairDipole(pybind11::module &m);
void export_AnisoPotentialPairGB(pybind11::module &m);

void export_PotentialBondHarmonic(pybind11::module& m);
void export_PotentialBondFENE(pybind11::module& m);
void export_PotentialBondTether(pybind11::module& m);

void export_IntegratorTwoStep(pybind11::module& m);
void export_IntegrationMethodTwoStep(pybind11::module& m);
void export_ZeroMomentumUpdater(pybind11::module& m);
void export_TwoStepNVE(pybind11::module& m);
void export_TwoStepNVTMTK(pybind11::module& m);
void export_TwoStepLangevinBase(pybind11::module& m);
void export_TwoStepLangevin(pybind11::module& m);
void export_TwoStepBD(pybind11::module& m);
void export_TwoStepNPTMTK(pybind11::module& m);
void export_Berendsen(pybind11::module& m);
void export_FIREEnergyMinimizer(pybind11::module& m);
void export_MuellerPlatheFlow(pybind11::module& m);

void export_TwoStepRATTLEBDCylinder(pybind11::module &m);
void export_TwoStepRATTLEBDDiamond(pybind11::module &m);
void export_TwoStepRATTLEBDEllipsoid(pybind11::module &m);
void export_TwoStepRATTLEBDGyroid(pybind11::module &m);
void export_TwoStepRATTLEBDPlane(pybind11::module &m);
void export_TwoStepRATTLEBDPrimitive(pybind11::module &m);
void export_TwoStepRATTLEBDSphere(pybind11::module &m);

void export_TwoStepRATTLELangevinCylinder(pybind11::module &m);
void export_TwoStepRATTLELangevinDiamond(pybind11::module &m);
void export_TwoStepRATTLELangevinEllipsoid(pybind11::module &m);
void export_TwoStepRATTLELangevinGyroid(pybind11::module &m);
void export_TwoStepRATTLELangevinPlane(pybind11::module &m);
void export_TwoStepRATTLELangevinPrimitive(pybind11::module &m);
void export_TwoStepRATTLELangevinSphere(pybind11::module &m);

void export_TwoStepRATTLENVECylinder(pybind11::module &m);
void export_TwoStepRATTLENVEDiamond(pybind11::module &m);
void export_TwoStepRATTLENVEEllipsoid(pybind11::module &m);
void export_TwoStepRATTLENVEGyroid(pybind11::module &m);
void export_TwoStepRATTLENVEPlane(pybind11::module &m);
void export_TwoStepRATTLENVEPrimitive(pybind11::module &m);
void export_TwoStepRATTLENVESphere(pybind11::module &m);

void export_ManifoldDiamond(pybind11::module& m);
void export_ManifoldEllipsoid(pybind11::module& m);
void export_ManifoldGyroid(pybind11::module& m);
void export_ManifoldPrimitive(pybind11::module& m);
void export_ManifoldSphere(pybind11::module& m);
void export_ManifoldXYPlane(pybind11::module& m);
void export_ManifoldZCylinder(pybind11::module& m);

#ifdef ENABLE_HIP

void export_ActiveForceConstraintComputeCylinderGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeDiamondGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeEllipsoidGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeGyroidGPU(pybind11::module &m);
void export_ActiveForceConstraintComputePlaneGPU(pybind11::module &m);
void export_ActiveForceConstraintComputePrimitiveGPU(pybind11::module &m);
void export_ActiveForceConstraintComputeSphereGPU(pybind11::module &m);
void export_ActiveForceComputeGPU(pybind11::module &m);
void export_ComputeThermoGPU(pybind11::module& m);
void export_ComputeThermoHMAGPU(pybind11::module& m);
void export_HarmonicAngleForceComputeGPU(pybind11::module& m);
void export_CosineSqAngleForceComputeGPU(pybind11::module& m);
void export_TableAngleForceComputeGPU(pybind11::module& m);
void export_HarmonicDihedralForceComputeGPU(pybind11::module& m);
void export_OPLSDihedralForceComputeGPU(pybind11::module& m);
void export_TableDihedralForceComputeGPU(pybind11::module& m);
void export_HarmonicImproperForceComputeGPU(pybind11::module& m);
void export_BondTablePotentialGPU(pybind11::module& m);
void export_NeighborListGPU(pybind11::module& m);
void export_NeighborListGPUBinned(pybind11::module& m);
void export_NeighborListGPUStencil(pybind11::module& m);
void export_NeighborListGPUTree(pybind11::module& m);
void export_ForceDistanceConstraintGPU(pybind11::module& m);
void export_ForceCompositeGPU(pybind11::module& m);
void export_PPPMForceComputeGPU(pybind11::module& m);

void export_AnisoPotentialPairALJ2DGPU(pybind11::module &m);
void export_AnisoPotentialPairALJ3DGPU(pybind11::module &m);
void export_AnisoPotentialPairDipoleGPU(pybind11::module &m);
void export_AnisoPotentialPairGBGPU(pybind11::module &m);

void export_PotentialBondHarmonicGPU(pybind11::module& m);
void export_PotentialBondFENEGPU(pybind11::module& m);
void export_PotentialBondTetherGPU(pybind11::module& m);

void export_TwoStepNVEGPU(pybind11::module& m);
void export_TwoStepNVTMTKGPU(pybind11::module& m);
void export_TwoStepLangevinGPU(pybind11::module& m);
void export_TwoStepBDGPU(pybind11::module& m);
void export_TwoStepNPTMTKGPU(pybind11::module& m);
void export_BerendsenGPU(pybind11::module& m);
void export_FIREEnergyMinimizerGPU(pybind11::module& m);
void export_MuellerPlatheFlowGPU(pybind11::module& m);

void export_TwoStepRATTLEBDGPUZCylinder(pybind11::module& m);
void export_TwoStepRATTLEBDGPUDiamond(pybind11::module& m);
void export_TwoStepRATTLEBDGPUEllipsoid(pybind11::module& m);
void export_TwoStepRATTLEBDGPUGyroid(pybind11::module& m);
void export_TwoStepRATTLEBDGPUXYPlane(pybind11::module& m);
void export_TwoStepRATTLEBDGPUPrimitive(pybind11::module& m);
void export_TwoStepRATTLEBDGPUSphere(pybind11::module& m);

void export_TwoStepRATTLELangevinGPUZCylinder(pybind11::module& m);
void export_TwoStepRATTLELangevinGPUDiamond(pybind11::module& m);
void export_TwoStepRATTLELangevinGPUEllipsoid(pybind11::module& m);
void export_TwoStepRATTLELangevinGPUGyroid(pybind11::module& m);
void export_TwoStepRATTLELangevinGPUXYPlane(pybind11::module& m);
void export_TwoStepRATTLELangevinGPUPrimitive(pybind11::module& m);
void export_TwoStepRATTLELangevinGPUSphere(pybind11::module& m);

void export_TwoStepRATTLENVEGPUZCylinder(pybind11::module& m);
void export_TwoStepRATTLENVEGPUDiamond(pybind11::module& m);
void export_TwoStepRATTLENVEGPUEllipsoid(pybind11::module& m);
void export_TwoStepRATTLENVEGPUGyroid(pybind11::module& m);
void export_TwoStepRATTLENVEGPUXYPlane(pybind11::module& m);
void export_TwoStepRATTLENVEGPUPrimitive(pybind11::module& m);
void export_TwoStepRATTLENVEGPUSphere(pybind11::module& m);
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
    export_PotentialPair<PotentialPairMie>(m, "PotentialPairMie");
    export_PotentialPair<PotentialPairReactionField>(m, "PotentialPairReactionField");
    export_PotentialPair<PotentialPairDLVO>(m, "PotentialPairDLVO");
    export_PotentialPair<PotentialPairFourier>(m, "PotentialPairFourier");
    export_PotentialPair<PotentialPairOPP>(m, "PotentialPairOPP");
    export_PotentialPair<PotentialPairTWF>(m, "PotentialPairTWF");
    export_PotentialPair<PotentialPairForceShiftedLJ>(m, "PotentialPairForceShiftedLJ");

    export_PotentialTersoff<PotentialTripletTersoff>(m, "PotentialTersoff");
    export_PotentialTersoff<PotentialTripletSquareDensity>(m, "PotentialSquareDensity");
    export_PotentialTersoff<PotentialTripletRevCross>(m, "PotentialRevCross");

    export_AnisoPotentialPairALJ2D(m);
    export_AnisoPotentialPairALJ3D(m);
    export_AnisoPotentialPairDipole(m);
    export_AnisoPotentialPairGB(m);

    export_PotentialPairDPDThermo<PotentialPairDPDThermoDPD, PotentialPairDPD>(
        m,
        "PotentialPairDPDThermoDPD");
    export_PotentialPair<PotentialPairDPDLJ>(m, "PotentialPairDPDLJ");
    export_PotentialPair<PotentialPairTable>(m, "PotentialPairTable");
    export_PotentialPairDPDThermo<PotentialPairDPDLJThermoDPD, PotentialPairDPDLJ>(
        m,
        "PotentialPairDPDLJThermoDPD");

    export_PotentialBondHarmonic(m);
    export_PotentialBondFENE(m);
    export_PotentialBondTether(m);

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

    export_PotentialBondHarmonicGPU(m);
    export_PotentialBondFENEGPU(m);
    export_PotentialBondTetherGPU(m);

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
    export_TwoStepRATTLEBDCylinder(m);
    export_TwoStepRATTLEBDDiamond(m);
    export_TwoStepRATTLEBDEllipsoid(m);
    export_TwoStepRATTLEBDGyroid(m);
    export_TwoStepRATTLEBDPlane(m);
    export_TwoStepRATTLEBDPrimitive(m);
    export_TwoStepRATTLEBDSphere(m);

    export_TwoStepRATTLELangevinCylinder(m);
    export_TwoStepRATTLELangevinDiamond(m);
    export_TwoStepRATTLELangevinEllipsoid(m);
    export_TwoStepRATTLELangevinGyroid(m);
    export_TwoStepRATTLELangevinPlane(m);
    export_TwoStepRATTLELangevinPrimitive(m);
    export_TwoStepRATTLELangevinSphere(m);

    export_TwoStepRATTLENVECylinder(m);
    export_TwoStepRATTLENVEDiamond(m);
    export_TwoStepRATTLENVEEllipsoid(m);
    export_TwoStepRATTLENVEGyroid(m);
    export_TwoStepRATTLENVEPlane(m);
    export_TwoStepRATTLENVEPrimitive(m);
    export_TwoStepRATTLENVESphere(m);

#ifdef ENABLE_HIP
    export_TwoStepNVEGPU(m);
    export_TwoStepNVTMTKGPU(m);
    export_TwoStepLangevinGPU(m);
    export_TwoStepBDGPU(m);
    export_TwoStepNPTMTKGPU(m);
    export_BerendsenGPU(m);
    export_FIREEnergyMinimizerGPU(m);
    export_MuellerPlatheFlowGPU(m);

    export_TwoStepRATTLEBDGPUZCylinder(m);
    export_TwoStepRATTLEBDGPUDiamond(m);
    export_TwoStepRATTLEBDGPUEllipsoid(m);
    export_TwoStepRATTLEBDGPUGyroid(m);
    export_TwoStepRATTLEBDGPUXYPlane(m);
    export_TwoStepRATTLEBDGPUPrimitive(m);
    export_TwoStepRATTLEBDGPUSphere(m);

    export_TwoStepRATTLELangevinGPUZCylinder(m);
    export_TwoStepRATTLELangevinGPUDiamond(m);
    export_TwoStepRATTLELangevinGPUEllipsoid(m);
    export_TwoStepRATTLELangevinGPUGyroid(m);
    export_TwoStepRATTLELangevinGPUXYPlane(m);
    export_TwoStepRATTLELangevinGPUPrimitive(m);
    export_TwoStepRATTLELangevinGPUSphere(m);

    export_TwoStepRATTLENVEGPUZCylinder(m);
    export_TwoStepRATTLENVEGPUDiamond(m);
    export_TwoStepRATTLENVEGPUEllipsoid(m);
    export_TwoStepRATTLENVEGPUGyroid(m);
    export_TwoStepRATTLENVEGPUXYPlane(m);
    export_TwoStepRATTLENVEGPUPrimitive(m);
    export_TwoStepRATTLENVEGPUSphere(m);
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
