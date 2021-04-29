// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander All developers are free to add the calls needed to export their modules

#include "ActiveForceCompute.h"
#include "AllAnisoPairPotentials.h"
#include "AllBondPotentials.h"
#include "AllExternalPotentials.h"
#include "AllPairPotentials.h"
#include "AllTripletPotentials.h"
#include "AllSpecialPairPotentials.h"
#include "AnisoPotentialPair.h"
#include "BondTablePotential.h"
#include "ComputeThermo.h"
#include "ComputeThermoHMA.h"
#include "ConstExternalFieldDipoleForceCompute.h"
#include "ConstraintEllipsoid.h"
#include "ConstraintSphere.h"
#include "OneDConstraint.h"
#include "Enforce2DUpdater.h"
#include "EvaluatorTersoff.h"
#include "EvaluatorSquareDensity.h"
#include "EvaluatorRevCross.h"
#include "FIREEnergyMinimizer.h"
#include "ForceComposite.h"
#include "ForceDistanceConstraint.h"
#include "HarmonicAngleForceCompute.h"
#include "CosineSqAngleForceCompute.h"
#include "HarmonicDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"
#include "IntegrationMethodTwoStep.h"
#include "IntegratorTwoStep.h"
#include "MolecularForceCompute.h"
#include "NeighborListBinned.h"
#include "NeighborList.h"
#include "NeighborListStencil.h"
#include "NeighborListTree.h"
#include "OPLSDihedralForceCompute.h"
#include "PotentialBond.h"
#include "PotentialExternal.h"
#include "PotentialPairDPDThermo.h"
#include "PotentialPair.h"
#include "PotentialTersoff.h"
#include "PPPMForceCompute.h"
#include "QuaternionMath.h"
#include "TableAngleForceCompute.h"
#include "TableDihedralForceCompute.h"
#include "TablePotential.h"
#include "TempRescaleUpdater.h"
#include "TwoStepBD.h"
#include "TwoStepBerendsen.h"
#include "TwoStepLangevinBase.h"
#include "TwoStepLangevin.h"
#include "TwoStepNPTMTK.h"
#include "TwoStepNVE.h"
#include "TwoStepNVTMTK.h"
#include "WallData.h"
#include "ZeroMomentumUpdater.h"
#include "MuellerPlatheFlow.h"

// include GPU classes
#ifdef ENABLE_HIP
#include "ActiveForceComputeGPU.h"
#include "AnisoPotentialPairGPU.h"
#include "BondTablePotentialGPU.h"
#include "ComputeThermoGPU.h"
#include "ComputeThermoHMAGPU.h"
#include "ConstraintEllipsoidGPU.h"
#include "ConstraintSphereGPU.h"
#include "OneDConstraintGPU.h"
#include "Enforce2DUpdaterGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#include "ForceCompositeGPU.h"
#include "ForceDistanceConstraintGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#include "CosineSqAngleForceComputeGPU.h"
#include "HarmonicDihedralForceComputeGPU.h"
#include "HarmonicImproperForceComputeGPU.h"
#include "NeighborListGPUBinned.h"
#include "NeighborListGPU.h"
#include "NeighborListGPUStencil.h"
#include "NeighborListGPUTree.h"
#include "OPLSDihedralForceComputeGPU.h"
#include "PotentialBondGPU.h"
#include "PotentialExternalGPU.h"
#include "PotentialPairDPDThermoGPU.h"
#include "PotentialPairGPU.h"
#include "PotentialTersoffGPU.h"
#include "PPPMForceComputeGPU.h"
#include "TableAngleForceComputeGPU.h"
#include "TableDihedralForceComputeGPU.h"
#include "TablePotentialGPU.h"
#include "TwoStepBDGPU.h"
#include "TwoStepBerendsenGPU.h"
#include "TwoStepLangevinGPU.h"
#include "TwoStepNPTMTKGPU.h"
#include "TwoStepNVEGPU.h"
#include "TwoStepNVTMTKGPU.h"
#include "MuellerPlatheFlowGPU.h"
#endif

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

//! Helper function for converting python wall group structure to wall_type
wall_type make_wall_field_params(py::object walls, std::shared_ptr<const ExecutionConfiguration> m_exec_conf)
    {
    wall_type w;
    py::list walls_spheres = walls.attr("spheres").cast<py::list>();
    py::list walls_cylinders = walls.attr("cylinders").cast<py::list>();
    py::list walls_planes = walls.attr("planes").cast<py::list>();
    w.numSpheres = (unsigned int)py::len(walls_spheres);
    w.numCylinders = (unsigned int)py::len(walls_cylinders);
    w.numPlanes = (unsigned int)py::len(walls_planes);

    if (w.numSpheres>MAX_N_SWALLS || w.numCylinders>MAX_N_CWALLS || w.numPlanes>MAX_N_PWALLS)
        {
        m_exec_conf->msg->error() << "A number of walls greater than the maximum number allowed was specified in a wall force." << std::endl;
        throw std::runtime_error("Error loading wall group.");
        }
    else
        {

        for(unsigned int i = 0; i < w.numSpheres; i++)
            {
            Scalar     r = py::cast<Scalar>(py::object(walls_spheres[i]).attr("r"));
            Scalar3 origin =py::cast<Scalar3>(py::object(walls_spheres[i]).attr("_origin"));
            bool     inside =py::cast<bool>(py::object(walls_spheres[i]).attr("inside"));
            w.Spheres[i] = SphereWall(r, origin, inside);
            }
        for(unsigned int i = 0; i < w.numCylinders; i++)
            {
            Scalar     r = py::cast<Scalar>(py::object(walls_cylinders[i]).attr("r"));
            Scalar3 origin =py::cast<Scalar3>(py::object(walls_cylinders[i]).attr("_origin"));
            Scalar3 axis =py::cast<Scalar3>(py::object(walls_cylinders[i]).attr("_axis"));
            bool     inside =py::cast<bool>(py::object(walls_cylinders[i]).attr("inside"));
            w.Cylinders[i] = CylinderWall(r, origin, axis, inside);
            }
        for(unsigned int i = 0; i < w.numPlanes; i++)
            {
            Scalar3 origin =py::cast<Scalar3>(py::object(walls_planes[i]).attr("_origin"));
            Scalar3 normal =py::cast<Scalar3>(py::object(walls_planes[i]).attr("_normal"));
            bool    inside =py::cast<bool>(py::object(walls_planes[i]).attr("inside"));
            w.Planes[i] = PlaneWall(origin, normal, inside);
            }
        return w;
        }
    }

//! Exports helper function for parameters based on standard evaluators
template< class evaluator >
void export_wall_params_helpers(py::module& m)
    {
    py::class_<typename EvaluatorWalls<evaluator>::param_type , std::shared_ptr<typename EvaluatorWalls<evaluator>::param_type> >(m, (EvaluatorWalls<evaluator>::getName()+"_params").c_str())
        .def(py::init<>())
        .def_readwrite("params", &EvaluatorWalls<evaluator>::param_type::params)
        .def_readwrite("rextrap", &EvaluatorWalls<evaluator>::param_type::rextrap)
        .def_readwrite("rcutsq", &EvaluatorWalls<evaluator>::param_type::rcutsq)
        ;
    m.def(std::string("make_"+EvaluatorWalls<evaluator>::getName()+"_params").c_str(), &make_wall_params<evaluator>);
    }

//! Combines exports of evaluators and parameter helper functions
template < class evaluator >
void export_PotentialExternalWall(py::module& m, const std::string& name)
    {
    export_PotentialExternal< PotentialExternal<EvaluatorWalls<evaluator> > >(m, name);
    export_wall_params_helpers<evaluator>(m);
    }


// Template specification for Dipole anisotropic pair potential. A specific
// template instance is needed since we expose the shape as just mu in Python
// when the default behavior exposes setting and getting the shape through
// 'shape'.
template<>
void export_AnisoPotentialPair<AnisoPotentialPairDipole>(
    pybind11::module& m, const std::string& name)
    {
    pybind11::class_<AnisoPotentialPairDipole, ForceCompute,
                     std::shared_ptr<AnisoPotentialPairDipole>
                     > anisopotentialpair(m, name.c_str());
    anisopotentialpair.def(
        pybind11::init<std::shared_ptr<SystemDefinition>,
                       std::shared_ptr<NeighborList>,
                       const std::string& >())
        .def("setParams", &AnisoPotentialPairDipole::setParamsPython)
        .def("getParams", &AnisoPotentialPairDipole::getParamsPython)
        .def("setMu", &AnisoPotentialPairDipole::setShapePython)
        .def("getMu", &AnisoPotentialPairDipole::getShapePython)
        .def("setRCut", &AnisoPotentialPairDipole::setRCutPython)
        .def("getRCut", &AnisoPotentialPairDipole::getRCut)
        .def_property("mode",
                      &AnisoPotentialPairDipole::getShiftMode,
                      &AnisoPotentialPairDipole::setShiftModePython)
        .def("slotWriteGSDShapeSpec",
             &AnisoPotentialPairDipole::slotWriteGSDShapeSpec)
        .def("connectGSDShapeSpec",
             &AnisoPotentialPairDipole::connectGSDShapeSpec)
        .def("getTypeShapesPy", &AnisoPotentialPairDipole::getTypeShapesPy)
    ;
    }


//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
PYBIND11_MODULE(_md, m)
    {
    export_ActiveForceCompute(m);
    export_ConstExternalFieldDipoleForceCompute(m);
    export_ComputeThermo(m);
    export_ComputeThermoHMA(m);
    export_HarmonicAngleForceCompute(m);
    export_CosineSqAngleForceCompute(m);
    export_TableAngleForceCompute(m);
    export_HarmonicDihedralForceCompute(m);
    export_OPLSDihedralForceCompute(m);
    export_TableDihedralForceCompute(m);
    export_HarmonicImproperForceCompute(m);
    export_TablePotential(m);
    export_BondTablePotential(m);
    export_PotentialPair<PotentialPairBuckingham>(m, "PotentialPairBuckingham");
    export_PotentialPair<PotentialPairLJ>(m, "PotentialPairLJ");
    export_PotentialPair<PotentialPairLJ1208>(m, "PotentialPairLJ1208");
    export_PotentialPair<PotentialPairLJ0804>(m, "PotentialPairLJ0804");
    export_PotentialPair<PotentialPairGauss>(m, "PotentialPairGauss");
    export_PotentialPair<PotentialPairSLJ>(m, "PotentialPairSLJ");
    export_PotentialPair<PotentialPairYukawa>(m, "PotentialPairYukawa");
    export_PotentialPair<PotentialPairEwald>(m, "PotentialPairEwald");
    export_PotentialPair<PotentialPairMorse>(m, "PotentialPairMorse");
    export_PotentialPair<PotentialPairDPD>(m, "PotentialPairDPD");
    export_PotentialPair<PotentialPairMoliere>(m, "PotentialPairMoliere");
    export_PotentialPair<PotentialPairZBL>(m, "PotentialPairZBL");
    export_PotentialTersoff<PotentialTripletTersoff>(m, "PotentialTersoff");
    export_PotentialTersoff<PotentialTripletSquareDensity> (m, "PotentialSquareDensity");
    export_PotentialTersoff<PotentialTripletRevCross> (m, "PotentialRevCross");
    export_PotentialPair<PotentialPairMie>(m, "PotentialPairMie");
    export_PotentialPair<PotentialPairReactionField>(m, "PotentialPairReactionField");
    export_PotentialPair<PotentialPairDLVO>(m, "PotentialPairDLVO");
    export_PotentialPair<PotentialPairFourier>(m, "PotentialPairFourier");
    export_PotentialPair<PotentialPairOPP>(m, "PotentialPairOPP");
    export_PotentialPair<PotentialPairTWF>(m, "PotentialPairTWF");
    export_AnisoPotentialPair<AnisoPotentialPairGB>(m, "AnisoPotentialPairGB");
    export_AnisoPotentialPair<AnisoPotentialPairDipole>(m, "AnisoPotentialPairDipole");
    export_PotentialPair<PotentialPairForceShiftedLJ>(m, "PotentialPairForceShiftedLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDThermoDPD, PotentialPairDPD>(m, "PotentialPairDPDThermoDPD");
    export_PotentialPair<PotentialPairDPDLJ>(m, "PotentialPairDPDLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDLJThermoDPD, PotentialPairDPDLJ>(m, "PotentialPairDPDLJThermoDPD");
    export_PotentialBond<PotentialBondHarmonic>(m, "PotentialBondHarmonic");
    export_PotentialBond<PotentialBondFENE>(m, "PotentialBondFENE");
    export_PotentialSpecialPair<PotentialSpecialPairLJ>(m, "PotentialSpecialPairLJ");
    export_PotentialSpecialPair<PotentialSpecialPairCoulomb>(m, "PotentialSpecialPairCoulomb");
    export_NeighborList(m);
    export_NeighborListBinned(m);
    export_NeighborListStencil(m);
    export_NeighborListTree(m);
    export_ConstraintSphere(m);
    export_OneDConstraint(m);
    export_MolecularForceCompute(m);
    export_ForceDistanceConstraint(m);
    export_ForceComposite(m);
    export_PPPMForceCompute(m);
    py::class_< wall_type, std::shared_ptr<wall_type> >(m, "wall_type")
        .def(py::init<>());
    m.def("make_wall_field_params", &make_wall_field_params);
    export_PotentialExternal<PotentialExternalPeriodic>(m, "PotentialExternalPeriodic");
    export_PotentialExternal<PotentialExternalElectricField>(m, "PotentialExternalElectricField");
    export_PotentialExternalWall<EvaluatorPairLJ>(m, "WallsPotentialLJ");
    export_PotentialExternalWall<EvaluatorPairYukawa>(m, "WallsPotentialYukawa");
    export_PotentialExternalWall<EvaluatorPairSLJ>(m, "WallsPotentialSLJ");
    export_PotentialExternalWall<EvaluatorPairForceShiftedLJ>(m, "WallsPotentialForceShiftedLJ");
    export_PotentialExternalWall<EvaluatorPairMie>(m, "WallsPotentialMie");
    export_PotentialExternalWall<EvaluatorPairGauss>(m, "WallsPotentialGauss");
    export_PotentialExternalWall<EvaluatorPairMorse>(m, "WallsPotentialMorse");

#ifdef ENABLE_HIP
    export_NeighborListGPU(m);
    export_NeighborListGPUBinned(m);
    export_NeighborListGPUStencil(m);
    export_NeighborListGPUTree(m);
    export_ForceCompositeGPU(m);
    export_PotentialPairGPU<PotentialPairBuckinghamGPU, PotentialPairBuckingham>(m, "PotentialPairBuckinghamGPU");
    export_PotentialPairGPU<PotentialPairLJGPU, PotentialPairLJ>(m, "PotentialPairLJGPU");
    export_PotentialPairGPU<PotentialPairLJ1208GPU, PotentialPairLJ1208>(m, "PotentialPairLJ1208GPU");
    export_PotentialPairGPU<PotentialPairLJ0804GPU, PotentialPairLJ0804>(m, "PotentialPairLJ0804GPU");
    export_PotentialPairGPU<PotentialPairGaussGPU, PotentialPairGauss>(m, "PotentialPairGaussGPU");
    export_PotentialPairGPU<PotentialPairSLJGPU, PotentialPairSLJ>(m, "PotentialPairSLJGPU");
    export_PotentialPairGPU<PotentialPairYukawaGPU, PotentialPairYukawa>(m, "PotentialPairYukawaGPU");
    export_PotentialPairGPU<PotentialPairReactionFieldGPU, PotentialPairReactionField>(m, "PotentialPairReactionFieldGPU");
    export_PotentialPairGPU<PotentialPairDLVOGPU, PotentialPairDLVO>(m, "PotentialPairDLVOGPU");
    export_PotentialPairGPU<PotentialPairFourierGPU, PotentialPairFourier>(m, "PotentialPairFourierGPU");
    export_PotentialPairGPU<PotentialPairEwaldGPU, PotentialPairEwald>(m, "PotentialPairEwaldGPU");
    export_PotentialPairGPU<PotentialPairMorseGPU, PotentialPairMorse>(m, "PotentialPairMorseGPU");
    export_PotentialPairGPU<PotentialPairDPDGPU, PotentialPairDPD>(m, "PotentialPairDPDGPU");
    export_PotentialPairGPU<PotentialPairMoliereGPU, PotentialPairMoliere>(m, "PotentialPairMoliereGPU");
    export_PotentialPairGPU<PotentialPairZBLGPU, PotentialPairZBL>(m, "PotentialPairZBLGPU");
    export_PotentialTersoffGPU<PotentialTripletTersoffGPU, PotentialTripletTersoff>(m, "PotentialTersoffGPU");
    export_PotentialTersoffGPU<PotentialTripletSquareDensityGPU, PotentialTripletSquareDensity> (m, "PotentialSquareDensityGPU");
    export_PotentialTersoffGPU<PotentialTripletRevCrossGPU, PotentialTripletRevCross> (m, "PotentialRevCrossGPU");
    export_PotentialPairGPU<PotentialPairForceShiftedLJGPU, PotentialPairForceShiftedLJ>(m, "PotentialPairForceShiftedLJGPU");
    export_PotentialPairGPU<PotentialPairMieGPU, PotentialPairMie>(m, "PotentialPairMieGPU");
    export_PotentialPairGPU<PotentialPairOPPGPU, PotentialPairOPP>(m, "PotentialPairOPPGPU");
    export_PotentialPairGPU<PotentialPairTWFGPU, PotentialPairTWF>(m, "PotentialPairTWFGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDThermoDPDGPU, PotentialPairDPDThermoDPD >(m, "PotentialPairDPDThermoDPDGPU");
    export_PotentialPairGPU<PotentialPairDPDLJGPU, PotentialPairDPDLJ>(m, "PotentialPairDPDLJGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDLJThermoDPDGPU, PotentialPairDPDLJThermoDPD >(m, "PotentialPairDPDLJThermoDPDGPU");
    export_AnisoPotentialPairGPU<AnisoPotentialPairGBGPU, AnisoPotentialPairGB>(m, "AnisoPotentialPairGBGPU");
    export_AnisoPotentialPairGPU<AnisoPotentialPairDipoleGPU, AnisoPotentialPairDipole>(m, "AnisoPotentialPairDipoleGPU");
    export_PotentialBondGPU<PotentialBondHarmonicGPU, PotentialBondHarmonic>(m, "PotentialBondHarmonicGPU");
    export_PotentialBondGPU<PotentialBondFENEGPU, PotentialBondFENE>(m, "PotentialBondFENEGPU");
    export_PotentialSpecialPairGPU<PotentialSpecialPairLJGPU, PotentialSpecialPairLJ>(m, "PotentialSpecialPairLJGPU");
    export_PotentialSpecialPairGPU<PotentialSpecialPairCoulombGPU, PotentialSpecialPairCoulomb>(m, "PotentialSpecialPairCoulombGPU");
    export_BondTablePotentialGPU(m);
    export_TablePotentialGPU(m);
    export_HarmonicAngleForceComputeGPU(m);
    export_CosineSqAngleForceComputeGPU(m);
    export_TableAngleForceComputeGPU(m);
    export_HarmonicDihedralForceComputeGPU(m);
    export_OPLSDihedralForceComputeGPU(m);
    export_TableDihedralForceComputeGPU(m);
    export_HarmonicImproperForceComputeGPU(m);
    export_ConstraintSphereGPU(m);
    export_OneDConstraintGPU(m);
    export_ForceDistanceConstraintGPU(m);
    // export_ConstExternalFieldDipoleForceComputeGPU(m);
    export_ComputeThermoGPU(m);
    export_ComputeThermoHMAGPU(m);
    export_PPPMForceComputeGPU(m);
    export_ActiveForceComputeGPU(m);
    export_PotentialExternalGPU<PotentialExternalPeriodicGPU, PotentialExternalPeriodic>(m, "PotentialExternalPeriodicGPU");
    export_PotentialExternalGPU<PotentialExternalElectricFieldGPU, PotentialExternalElectricField>(m, "PotentialExternalElectricFieldGPU");
    export_PotentialExternalGPU<WallsPotentialLJGPU, WallsPotentialLJ>(m, "WallsPotentialLJGPU");
    export_PotentialExternalGPU<WallsPotentialYukawaGPU, WallsPotentialYukawa>(m, "WallsPotentialYukawaGPU");
    export_PotentialExternalGPU<WallsPotentialSLJGPU, WallsPotentialSLJ>(m, "WallsPotentialSLJGPU");
    export_PotentialExternalGPU<WallsPotentialForceShiftedLJGPU, WallsPotentialForceShiftedLJ>(m, "WallsPotentialForceShiftedLJGPU");
    export_PotentialExternalGPU<WallsPotentialMieGPU, WallsPotentialMie>(m, "WallsPotentialMieGPU");
    export_PotentialExternalGPU<WallsPotentialGaussGPU, WallsPotentialGauss>(m, "WallsPotentialGaussGPU");
    export_PotentialExternalGPU<WallsPotentialMorseGPU, WallsPotentialMorse>(m, "WallsPotentialMorseGPU");
#endif

    // updaters
    export_IntegratorTwoStep(m);
    export_IntegrationMethodTwoStep(m);
    export_TempRescaleUpdater(m);
    export_ZeroMomentumUpdater(m);
    export_TwoStepNVE(m);
    export_TwoStepNVTMTK(m);
    export_TwoStepLangevinBase(m);
    export_TwoStepLangevin(m);
    export_TwoStepBD(m);
    export_TwoStepNPTMTK(m);
    export_Berendsen(m);
    export_Enforce2DUpdater(m);
    export_ConstraintEllipsoid(m);
    export_FIREEnergyMinimizer(m);
    export_MuellerPlatheFlow(m);

#ifdef ENABLE_HIP
    export_TwoStepNVEGPU(m);
    export_TwoStepNVTMTKGPU(m);
    export_TwoStepLangevinGPU(m);
    export_TwoStepBDGPU(m);
    export_TwoStepNPTMTKGPU(m);
    export_BerendsenGPU(m);
    export_Enforce2DUpdaterGPU(m);
    export_FIREEnergyMinimizerGPU(m);
    export_ConstraintEllipsoidGPU(m);
    export_MuellerPlatheFlowGPU(m);
#endif
    }
