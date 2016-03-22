/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander All developers are free to add the calls needed to export their modules

#include "ActiveForceCompute.h"
#include "AllAnisoPairPotentials.h"
#include "AllBondPotentials.h"
#include "AllExternalPotentials.h"
#include "AllPairPotentials.h"
#include "AllTripletPotentials.h"
#include "AnisoPotentialPair.h"
#include "BondTablePotential.h"
#include "CGCMMAngleForceCompute.h"
#include "CGCMMForceCompute.h"
#include "ConstExternalFieldDipoleForceCompute.h"
#include "ConstraintEllipsoid.h"
#include "ConstraintSphere.h"
#include "EAMForceCompute.h"
#include "Enforce2DUpdater.h"
#include "EvaluatorTersoff.h"
#include "FIREEnergyMinimizer.h"
#include "ForceComposite.h"
#include "ForceDistanceConstraint.h"
#include "HarmonicAngleForceCompute.h"
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

// include GPU classes
#ifdef ENABLE_CUDA
#include "ActiveForceComputeGPU.h"
#include "AnisoPotentialPairGPU.h"
#include "BondTablePotentialGPU.h"
#include "CGCMMAngleForceComputeGPU.h"
#include "CGCMMForceComputeGPU.h"
#include "ConstraintEllipsoidGPU.h"
#include "ConstraintSphereGPU.h"
#include "EAMForceComputeGPU.h"
#include "Enforce2DUpdaterGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#include "ForceCompositeGPU.h"
#include "ForceDistanceConstraintGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
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
#endif

#include <boost/python.hpp>

using namespace boost::python;

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

//! Function to export the tersoff parameter type to python
void export_tersoff_params()
{
    class_<tersoff_params>("tersoff_params", init<>())
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

    def("make_tersoff_params", &make_tersoff_params);
}

//! Helper function for converting python wall group structure to wall_type
wall_type make_wall_field_params(boost::python::object walls, boost::shared_ptr<const ExecutionConfiguration> m_exec_conf)
    {
    wall_type w;
    w.numSpheres = boost::python::len(walls.attr("spheres"));
    w.numCylinders = boost::python::len(walls.attr("cylinders"));
    w.numPlanes = boost::python::len(walls.attr("planes"));

    if (w.numSpheres>MAX_N_SWALLS || w.numCylinders>MAX_N_CWALLS || w.numPlanes>MAX_N_PWALLS)
        {
        m_exec_conf->msg->error() << "A number of walls greater than the maximum number allowed was specified in a wall force." << std::endl;
        throw std::runtime_error("Error loading wall group.");
        }
    else
        {
        for(unsigned int i = 0; i < w.numSpheres; i++)
            {
            Scalar     r = boost::python::extract<Scalar>(walls.attr("spheres")[i].attr("r"));
            Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("spheres")[i].attr("_origin"));
            bool     inside =boost::python::extract<bool>(walls.attr("spheres")[i].attr("inside"));
            w.Spheres[i] = SphereWall(r, origin, inside);
            }
        for(unsigned int i = 0; i < w.numCylinders; i++)
            {
            Scalar     r = boost::python::extract<Scalar>(walls.attr("cylinders")[i].attr("r"));
            Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("cylinders")[i].attr("_origin"));
            Scalar3 axis =boost::python::extract<Scalar3>(walls.attr("cylinders")[i].attr("_axis"));
            bool     inside =boost::python::extract<bool>(walls.attr("cylinders")[i].attr("inside"));
            w.Cylinders[i] = CylinderWall(r, origin, axis, inside);
            }
        for(unsigned int i = 0; i < w.numPlanes; i++)
            {
            Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("planes")[i].attr("_origin"));
            Scalar3 normal =boost::python::extract<Scalar3>(walls.attr("planes")[i].attr("_normal"));
            bool    inside =boost::python::extract<bool>(walls.attr("planes")[i].attr("inside"));
            w.Planes[i] = PlaneWall(origin, normal, inside);
            }
        return w;
        }
    }

//! Exports helper function for parameters based on standard evaluators
template< class evaluator >
void export_wall_params_helpers()
    {
    using namespace boost::python;
    class_<typename EvaluatorWalls<evaluator>::param_type , boost::shared_ptr<typename EvaluatorWalls<evaluator>::param_type> >((EvaluatorWalls<evaluator>::getName()+"_params").c_str(), init<>())
        .def_readwrite("params", &EvaluatorWalls<evaluator>::param_type::params)
        .def_readwrite("rextrap", &EvaluatorWalls<evaluator>::param_type::rextrap)
        .def_readwrite("rcutsq", &EvaluatorWalls<evaluator>::param_type::rcutsq)
        ;
    def(std::string("make_"+EvaluatorWalls<evaluator>::getName()+"_params").c_str(), &make_wall_params<evaluator>);

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION >= 106000)
    register_ptr_to_python< boost::shared_ptr<typename EvaluatorWalls<evaluator>::param_type > >();
    #endif
    }

//! Combines exports of evaluators and parameter helper functions
template < class evaluator >
void export_PotentialExternalWall(const std::string& name)
    {
    export_PotentialExternal< PotentialExternal<EvaluatorWalls<evaluator> > >(name);
    export_wall_params_helpers<evaluator>();
    }


//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(_md)
    {
    export_ActiveForceCompute();
    export_ConstExternalFieldDipoleForceCompute();
    export_HarmonicAngleForceCompute();
    export_TableAngleForceCompute();
    export_HarmonicDihedralForceCompute();
    export_OPLSDihedralForceCompute();
    export_TableDihedralForceCompute();
    export_HarmonicImproperForceCompute();
    export_CGCMMAngleForceCompute();
    export_TablePotential();
    export_BondTablePotential();
    export_CGCMMForceCompute();
    export_PotentialPair<PotentialPairLJ>("PotentialPairLJ");
    export_PotentialPair<PotentialPairGauss>("PotentialPairGauss");
    export_PotentialPair<PotentialPairSLJ>("PotentialPairSLJ");
    export_PotentialPair<PotentialPairYukawa>("PotentialPairYukawa");
    export_PotentialPair<PotentialPairEwald>("PotentialPairEwald");
    export_PotentialPair<PotentialPairMorse>("PotentialPairMorse");
    export_PotentialPair<PotentialPairDPD> ("PotentialPairDPD");
    export_PotentialPair<PotentialPairMoliere> ("PotentialPairMoliere");
    export_PotentialPair<PotentialPairZBL> ("PotentialPairZBL");
    export_PotentialTersoff<PotentialTripletTersoff> ("PotentialTersoff");
    export_PotentialPair<PotentialPairMie>("PotentialPairMie");
    export_PotentialPair<PotentialPairReactionField>("PotentialPairReactionField");
    export_tersoff_params();
    export_AnisoPotentialPair<AnisoPotentialPairGB> ("AnisoPotentialPairGB");
    export_AnisoPotentialPair<AnisoPotentialPairDipole> ("AnisoPotentialPairDipole");
    export_PotentialPair<PotentialPairForceShiftedLJ>("PotentialPairForceShiftedLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDThermoDPD, PotentialPairDPD>("PotentialPairDPDThermoDPD");
    export_PotentialPair<PotentialPairDPDLJ> ("PotentialPairDPDLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDLJThermoDPD, PotentialPairDPDLJ>("PotentialPairDPDLJThermoDPD");
    export_PotentialBond<PotentialBondHarmonic>("PotentialBondHarmonic");
    export_PotentialBond<PotentialBondFENE>("PotentialBondFENE");
    export_EAMForceCompute();
    export_NeighborList();
    export_NeighborListBinned();
    export_NeighborListStencil();
    export_NeighborListTree();
    export_ConstraintSphere();
    export_MolecularForceCompute();
    export_ForceDistanceConstraint();
    export_ForceComposite();
    export_PPPMForceCompute();
    class_< wall_type, boost::shared_ptr<wall_type> >( "wall_type", init<>());
    def("make_wall_field_params", &make_wall_field_params);
    export_PotentialExternal<PotentialExternalPeriodic>("PotentialExternalPeriodic");
    export_PotentialExternal<PotentialExternalElectricField>("PotentialExternalElectricField");
    export_PotentialExternalWall<EvaluatorPairLJ>("WallsPotentialLJ");
    export_PotentialExternalWall<EvaluatorPairYukawa>("WallsPotentialYukawa");
    export_PotentialExternalWall<EvaluatorPairSLJ>("WallsPotentialSLJ");
    export_PotentialExternalWall<EvaluatorPairForceShiftedLJ>("WallsPotentialForceShiftedLJ");
    export_PotentialExternalWall<EvaluatorPairMie>("WallsPotentialMie");
    export_PotentialExternalWall<EvaluatorPairGauss>("WallsPotentialGauss");
    export_PotentialExternalWall<EvaluatorPairMorse>("WallsPotentialMorse");

#ifdef ENABLE_CUDA
    export_NeighborListGPU();
    export_NeighborListGPUBinned();
    export_NeighborListGPUStencil();
    export_NeighborListGPUTree();
    export_CGCMMForceComputeGPU();
    export_ForceCompositeGPU();
    export_PotentialPairGPU<PotentialPairLJGPU, PotentialPairLJ>("PotentialPairLJGPU");
    export_PotentialPairGPU<PotentialPairGaussGPU, PotentialPairGauss>("PotentialPairGaussGPU");
    export_PotentialPairGPU<PotentialPairSLJGPU, PotentialPairSLJ>("PotentialPairSLJGPU");
    export_PotentialPairGPU<PotentialPairYukawaGPU, PotentialPairYukawa>("PotentialPairYukawaGPU");
    export_PotentialPairGPU<PotentialPairReactionFieldGPU, PotentialPairReactionField>("PotentialPairReactionFieldGPU");
    export_PotentialPairGPU<PotentialPairEwaldGPU, PotentialPairEwald>("PotentialPairEwaldGPU");
    export_PotentialPairGPU<PotentialPairMorseGPU, PotentialPairMorse>("PotentialPairMorseGPU");
    export_PotentialPairGPU<PotentialPairDPDGPU, PotentialPairDPD> ("PotentialPairDPDGPU");
    export_PotentialPairGPU<PotentialPairMoliereGPU, PotentialPairMoliere> ("PotentialPairMoliereGPU");
    export_PotentialPairGPU<PotentialPairZBLGPU, PotentialPairZBL> ("PotentialPairZBLGPU");
    export_PotentialTersoffGPU<PotentialTripletTersoffGPU, PotentialTripletTersoff> ("PotentialTersoffGPU");
    export_PotentialPairGPU<PotentialPairForceShiftedLJGPU, PotentialPairForceShiftedLJ>("PotentialPairForceShiftedLJGPU");
    export_PotentialPairGPU<PotentialPairMieGPU, PotentialPairMie>("PotentialPairMieGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDThermoDPDGPU, PotentialPairDPDThermoDPD >("PotentialPairDPDThermoDPDGPU");
    export_PotentialPairGPU<PotentialPairDPDLJGPU, PotentialPairDPDLJ> ("PotentialPairDPDLJGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDLJThermoDPDGPU, PotentialPairDPDLJThermoDPD >("PotentialPairDPDLJThermoDPDGPU");
    export_AnisoPotentialPairGPU<AnisoPotentialPairGBGPU, AnisoPotentialPairGB> ("AnisoPotentialPairGBGPU");
    export_AnisoPotentialPairGPU<AnisoPotentialPairDipoleGPU, AnisoPotentialPairDipole> ("AnisoPotentialPairDipoleGPU");
    export_PotentialBondGPU<PotentialBondHarmonicGPU, PotentialBondHarmonic>("PotentialBondHarmonicGPU");
    export_PotentialBondGPU<PotentialBondFENEGPU, PotentialBondFENE>("PotentialBondFENEGPU");
    export_BondTablePotentialGPU();
    export_TablePotentialGPU();
    export_EAMForceComputeGPU();
    export_HarmonicAngleForceComputeGPU();
    export_TableAngleForceComputeGPU();
    export_HarmonicDihedralForceComputeGPU();
    export_OPLSDihedralForceComputeGPU();
    export_TableDihedralForceComputeGPU();
    export_HarmonicImproperForceComputeGPU();
    export_CGCMMAngleForceComputeGPU();
    export_ConstraintSphereGPU();
    export_ForceDistanceConstraintGPU();
//    export_ConstExternalFieldDipoleForceComputeGPU();
    export_PPPMForceComputeGPU();
    export_ActiveForceComputeGPU();
    export_PotentialExternalGPU<PotentialExternalPeriodicGPU, PotentialExternalPeriodic>("PotentialExternalPeriodicGPU");
    export_PotentialExternalGPU<PotentialExternalElectricFieldGPU, PotentialExternalElectricField>("PotentialExternalElectricFieldGPU");
    export_PotentialExternalGPU<WallsPotentialLJGPU, WallsPotentialLJ>("WallsPotentialLJGPU");
    export_PotentialExternalGPU<WallsPotentialYukawaGPU, WallsPotentialYukawa>("WallsPotentialYukawaGPU");
    export_PotentialExternalGPU<WallsPotentialSLJGPU, WallsPotentialSLJ>("WallsPotentialSLJGPU");
    export_PotentialExternalGPU<WallsPotentialForceShiftedLJGPU, WallsPotentialForceShiftedLJ>("WallsPotentialForceShiftedLJGPU");
    export_PotentialExternalGPU<WallsPotentialMieGPU, WallsPotentialMie>("WallsPotentialMieGPU");
    export_PotentialExternalGPU<WallsPotentialGaussGPU, WallsPotentialGauss>("WallsPotentialGaussGPU");
    export_PotentialExternalGPU<WallsPotentialMorseGPU, WallsPotentialMorse>("WallsPotentialMorseGPU");
#endif

    // updaters
    export_IntegratorTwoStep();
    export_IntegrationMethodTwoStep();
    export_TempRescaleUpdater();
    export_ZeroMomentumUpdater();
    export_TwoStepNVE();
    export_TwoStepNVTMTK();
    export_TwoStepLangevinBase();
    export_TwoStepLangevin();
    export_TwoStepBD();
    export_TwoStepNPTMTK();
    export_Berendsen();
    export_Enforce2DUpdater();
    export_ConstraintEllipsoid();
    export_FIREEnergyMinimizer();

#ifdef ENABLE_CUDA
    export_TwoStepNVEGPU();
    export_TwoStepNVTMTKGPU();
    export_TwoStepLangevinGPU();
    export_TwoStepBDGPU();
    export_TwoStepNPTMTKGPU();
    export_BerendsenGPU();
    export_Enforce2DUpdaterGPU();
    export_FIREEnergyMinimizerGPU();
    export_ConstraintEllipsoidGPU();
#endif

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION >= 106000)
    register_ptr_to_python< boost::shared_ptr< IMDInterface > >();
    // register_ptr_to_python< boost::shared_ptr< AnalyzerWrap > >();
    register_ptr_to_python< boost::shared_ptr< DCDDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< POSDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< HOOMDDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< PDBDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< MOL2DumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< MSDAnalyzer > >();
    register_ptr_to_python< boost::shared_ptr< Logger > >();
    register_ptr_to_python< boost::shared_ptr< CallbackAnalyzer > >();
    register_ptr_to_python< boost::shared_ptr< DomainDecomposition > >();
    // register_ptr_to_python< boost::shared_ptr< ComputeWrap > >();
    register_ptr_to_python< boost::shared_ptr< TablePotential > >();
    register_ptr_to_python< boost::shared_ptr< PPPMForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< ConstExternalFieldDipoleForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< CellList > >();
    register_ptr_to_python< boost::shared_ptr< EAMForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMAngleForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< OPLSDihedralForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListStencil > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListBinned > >();
    register_ptr_to_python< boost::shared_ptr< ConstForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicDihedralForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< TableDihedralForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< CellListStencil > >();
    register_ptr_to_python< boost::shared_ptr< TableAngleForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< ConstraintSphere > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicImproperForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< NeighborList > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListTree > >();
    register_ptr_to_python< boost::shared_ptr< ForceConstraint > >();
    register_ptr_to_python< boost::shared_ptr< BondTablePotential > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicAngleForceCompute > >();
    // register_ptr_to_python< boost::shared_ptr< ForceComputeWrap > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< ExecutionConfiguration > >();
    register_ptr_to_python< boost::shared_ptr< SystemDefinition > >();
    register_ptr_to_python< boost::shared_ptr< ParticleData > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotParticleData<float> > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotParticleData<double> > >();
    register_ptr_to_python< boost::shared_ptr< RandomGenerator > >();
    // register_ptr_to_python< boost::shared_ptr< ParticleGeneratorWrap > >();
    register_ptr_to_python< boost::shared_ptr< PolymerParticleGenerator > >();
    register_ptr_to_python< boost::shared_ptr< HOOMDInitializer > >();
    register_ptr_to_python< boost::shared_ptr< ParticleGroup > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelector > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorAll > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorTag > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorType > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorRigid > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorCuboid > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotSystemData<float> > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotSystemData<double> > >();
    register_ptr_to_python< boost::shared_ptr< wall_type > >();
    register_ptr_to_python< boost::shared_ptr< System > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVE > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepLangevinBase > >();
    register_ptr_to_python< boost::shared_ptr< Enforce2DUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBD > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTMTK > >();
    register_ptr_to_python< boost::shared_ptr< BoxResizeUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TempRescaleUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPTMTK > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBerendsen > >();
    register_ptr_to_python< boost::shared_ptr< IntegratorTwoStep > >();
    // register_ptr_to_python< boost::shared_ptr< UpdaterWrap > >();
    register_ptr_to_python< boost::shared_ptr< Integrator > >();
    register_ptr_to_python< boost::shared_ptr< IntegrationMethodTwoStep > >();
    register_ptr_to_python< boost::shared_ptr< ZeroMomentumUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepLangevin > >();
    register_ptr_to_python< boost::shared_ptr< SFCPackUpdater > >();
    register_ptr_to_python< boost::shared_ptr< FIREEnergyMinimizer > >();
    register_ptr_to_python< boost::shared_ptr< double2 > >();
    register_ptr_to_python< boost::shared_ptr< double3 > >();
    register_ptr_to_python< boost::shared_ptr< double4 > >();
    register_ptr_to_python< boost::shared_ptr< float2 > >();
    register_ptr_to_python< boost::shared_ptr< float3 > >();
    register_ptr_to_python< boost::shared_ptr< float4 > >();
    register_ptr_to_python< boost::shared_ptr< uint2 > >();
    register_ptr_to_python< boost::shared_ptr< uint3 > >();
    register_ptr_to_python< boost::shared_ptr< uint4 > >();
    register_ptr_to_python< boost::shared_ptr< int2 > >();
    register_ptr_to_python< boost::shared_ptr< int3 > >();
    register_ptr_to_python< boost::shared_ptr< int4 > >();
    register_ptr_to_python< boost::shared_ptr< char3 > >();
    register_ptr_to_python< boost::shared_ptr< Variant > >();
    register_ptr_to_python< boost::shared_ptr< VariantConst > >();
    register_ptr_to_python< boost::shared_ptr< VariantLinear > >();
    register_ptr_to_python< boost::shared_ptr< Messenger > >();
    register_ptr_to_python< boost::shared_ptr< MolecularForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< ForceDistanceConstraint > >();
    register_ptr_to_python< boost::shared_ptr< ForceComposite > >();

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_MPI
    register_ptr_to_python< boost::shared_ptr< LoadBalancerGPU > >();
    register_ptr_to_python< boost::shared_ptr< CommunicatorGPU > >();
    #endif
    register_ptr_to_python< boost::shared_ptr< TableAngleForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicAngleForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListGPUStencil > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicImproperForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< PPPMForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< TableDihedralForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListGPU > >();
    register_ptr_to_python< boost::shared_ptr< TablePotentialGPU > >();
    register_ptr_to_python< boost::shared_ptr< BondTablePotentialGPU > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListGPUBinned > >();
    register_ptr_to_python< boost::shared_ptr< NeighborListGPUTree > >();
    register_ptr_to_python< boost::shared_ptr< ForceCompositeGPU > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicDihedralForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CellListGPU > >();
    register_ptr_to_python< boost::shared_ptr< ConstraintSphereGPU > >();
    register_ptr_to_python< boost::shared_ptr< OPLSDihedralForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMAngleForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< EAMForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepLangevinGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVEGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPTMTKGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTMTKGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBDGPU > >();
    register_ptr_to_python< boost::shared_ptr< FIREEnergyMinimizerGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBerendsenGPU > >();
    register_ptr_to_python< boost::shared_ptr< SFCPackUpdaterGPU > >();
    register_ptr_to_python< boost::shared_ptr< Enforce2DUpdaterGPU > >();
    register_ptr_to_python< boost::shared_ptr< ForceDistanceConstraintGPU > >();
    #endif

    #endif
    }
