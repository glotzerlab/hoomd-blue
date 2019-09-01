// Copyright (c) 2009-2019 The Regents of the University of Michigan
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
#include "ConstExternalFieldDipoleForceCompute.h"
#include "ConstraintEllipsoid.h"
#include "ConstraintSphere.h"
#include "OneDConstraint.h"
#include "Enforce2DUpdater.h"
#include "EvaluatorTersoff.h"
#include "EvaluatorSquareDensity.h"
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
#ifdef ENABLE_CUDA
#include "ActiveForceComputeGPU.h"
#include "AnisoPotentialPairGPU.h"
#include "BondTablePotentialGPU.h"
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

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
namespace py = pybind11;

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

//! Function to export the tersoff parameter type to python
void export_tersoff_params(py::module& m)
{
    py::class_<tersoff_params>(m, "tersoff_params")
        .def(py::init<>())
        .def_readwrite("cutoff_thickness", &tersoff_params::cutoff_thickness)
        .def_readwrite("coeffs", &tersoff_params::coeffs)
        .def_readwrite("exp_consts", &tersoff_params::exp_consts)
        .def_readwrite("dimer_r", &tersoff_params::dimer_r)
        .def_readwrite("tersoff_n", &tersoff_params::tersoff_n)
        .def_readwrite("gamman", &tersoff_params::gamman)
        .def_readwrite("lambda_cube", &tersoff_params::lambda_cube)
        .def_readwrite("ang_consts", &tersoff_params::ang_consts)
        .def_readwrite("alpha", &tersoff_params::alpha)
        ;

    m.def("make_tersoff_params", &make_tersoff_params);
}

//! Function to make the parameter type
pair_fourier_params make_pair_fourier_params(py::list a, py::list b)
    {
    pair_fourier_params retval;
    for (int i = 0; i < 3; ++i)
        {
        retval.a[i] = py::cast<Scalar>(a[i]);
        retval.b[i] = py::cast<Scalar>(b[i]);
        }
    return retval;
    }

// ! Function to export the fourier parameter type to python
void export_pair_fourier_params(py::module& m)
{
    py::class_<pair_fourier_params>(m, "pair_fourier_params")
        .def(py::init<>())
        ;

    m.def("make_pair_fourier_params", &make_pair_fourier_params);
}

//! Helper function for converting python wall group structure to wall_type
wall_type make_wall_field_params(py::object walls, std::shared_ptr<const ExecutionConfiguration> m_exec_conf)
    {
    wall_type w;
    py::list walls_spheres = walls.attr("spheres").cast<py::list>();
    py::list walls_cylinders = walls.attr("cylinders").cast<py::list>();
    py::list walls_planes = walls.attr("planes").cast<py::list>();
    w.numSpheres = py::len(walls_spheres);
    w.numCylinders = py::len(walls_cylinders);
    w.numPlanes = py::len(walls_planes);

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


//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
PYBIND11_MODULE(_md, m)
    {
    export_ActiveForceCompute(m);
    export_ConstExternalFieldDipoleForceCompute(m);
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
    export_PotentialPair<PotentialPairMie>(m, "PotentialPairMie");
    export_PotentialPair<PotentialPairReactionField>(m, "PotentialPairReactionField");
    export_PotentialPair<PotentialPairDLVO>(m, "PotentialPairDLVO");
    export_PotentialPair<PotentialPairFourier>(m, "PotentialPairFourier");
    export_tersoff_params(m);
    export_pair_fourier_params(m);
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

#ifdef ENABLE_CUDA
    export_NeighborListGPU(m);
    export_NeighborListGPUBinned(m);
    export_NeighborListGPUStencil(m);
    export_NeighborListGPUTree(m);
    export_ForceCompositeGPU(m);
    export_PotentialPairGPU<PotentialPairBuckinghamGPU, PotentialPairBuckingham>(m, "PotentialPairBuckinghamGPU");
    export_PotentialPairGPU<PotentialPairLJGPU, PotentialPairLJ>(m, "PotentialPairLJGPU");
    export_PotentialPairGPU<PotentialPairLJ1208GPU, PotentialPairLJ1208>(m, "PotentialPairLJ1208GPU");
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
    export_PotentialPairGPU<PotentialPairForceShiftedLJGPU, PotentialPairForceShiftedLJ>(m, "PotentialPairForceShiftedLJGPU");
    export_PotentialPairGPU<PotentialPairMieGPU, PotentialPairMie>(m, "PotentialPairMieGPU");
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

#ifdef ENABLE_CUDA
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
