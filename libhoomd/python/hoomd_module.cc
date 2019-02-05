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

// temporarily work around issues with the new boost fileystem libraries
// http://www.boost.org/doc/libs/1_46_1/libs/filesystem/v3/doc/index.htm
#include "HOOMDMath.h"
#include "ExecutionConfiguration.h"
#include "ClockSource.h"
#include "Profiler.h"
#include "ParticleData.h"
#include "RigidData.h"
#include "SystemDefinition.h"
#include "BondedGroupData.h"
#include "Initializers.h"
#include "HOOMDInitializer.h"
#include "RandomGenerator.h"
#include "Compute.h"
#include "CellList.h"
#include "CellListStencil.h"
#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "ConstForceCompute.h"
#include "ConstExternalFieldDipoleForceCompute.h"
#include "HarmonicAngleForceCompute.h"
#include "TableAngleForceCompute.h"
#include "HarmonicDihedralForceCompute.h"
#include "OPLSDihedralForceCompute.h"
#include "TableDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"
#include "CGCMMAngleForceCompute.h"
#include "CGCMMForceCompute.h"
#include "TablePotential.h"
#include "BondTablePotential.h"
#include "AllPairPotentials.h"
#include "AllBondPotentials.h"
#include "AllTripletPotentials.h"
#include "AllAnisoPairPotentials.h"
#include "ComputeThermo.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "NeighborListStencil.h"
#include "NeighborListTree.h"
#include "Analyzer.h"
#include "IMDInterface.h"
#include "HOOMDDumpWriter.h"
#include "POSDumpWriter.h"
#include "PDBDumpWriter.h"
#include "MOL2DumpWriter.h"
#include "DCDDumpWriter.h"
#include "Logger.h"
#include "MSDAnalyzer.h"
#include "CallbackAnalyzer.h"
#include "Updater.h"
#include "Integrator.h"
#include "IntegratorTwoStep.h"
#include "IntegrationMethodTwoStep.h"
#include "TwoStepNVE.h"
#include "TwoStepNVT.h"
#include "TwoStepNVTMTK.h"
#include "TwoStepLangevinBase.h"
#include "TwoStepLangevin.h"
#include "TwoStepBD.h"
#include "TwoStepNPTMTK.h"
#include "TwoStepBerendsen.h"
#include "TwoStepNHRigid.h"
#include "TwoStepNVERigid.h"
#include "TwoStepNVTRigid.h"
#include "TwoStepNPTRigid.h"
#include "TwoStepNPHRigid.h"
#include "TwoStepBDNVTRigid.h"
#include "TempRescaleUpdater.h"
#include "ZeroMomentumUpdater.h"
#include "FIREEnergyMinimizer.h"
#include "FIREEnergyMinimizerRigid.h"
#include "SFCPackUpdater.h"
#include "BoxResizeUpdater.h"
#include "Enforce2DUpdater.h"
#include "System.h"
#include "Variant.h"
#include "EAMForceCompute.h"
#include "ConstraintSphere.h"
#include "PotentialPairDPDThermo.h"
#include "EvaluatorTersoff.h"
#include "PotentialPair.h"
#include "PotentialTersoff.h"
#include "PPPMForceCompute.h"
#include "AllExternalPotentials.h"
#include "Messenger.h"

// include GPU classes
#ifdef ENABLE_CUDA
#include <cuda.h>
#include "CellListGPU.h"
#include "TwoStepNVEGPU.h"
#include "TwoStepNVTGPU.h"
#include "TwoStepLangevinGPU.h"
#include "TwoStepBDGPU.h"
#include "TwoStepNPTMTKGPU.h"
#include "TwoStepNVTMTKGPU.h"
#include "TwoStepBerendsenGPU.h"
#include "TwoStepNVERigidGPU.h"
#include "TwoStepNVTRigidGPU.h"
#include "TwoStepNPHRigidGPU.h"
#include "TwoStepNPTRigidGPU.h"
#include "TwoStepBDNVTRigidGPU.h"
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
#include "NeighborListGPUStencil.h"
#include "NeighborListGPUTree.h"
#include "CGCMMForceComputeGPU.h"
//#include "ConstExternalFieldDipoleForceComputeGPU.h"
#include "BondTablePotentialGPU.h"
#include "TablePotentialGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#include "TableAngleForceComputeGPU.h"
#include "HarmonicDihedralForceComputeGPU.h"
#include "OPLSDihedralForceComputeGPU.h"
#include "TableDihedralForceComputeGPU.h"
#include "HarmonicImproperForceComputeGPU.h"
#include "CGCMMAngleForceComputeGPU.h"
#include "Enforce2DUpdaterGPU.h"
#include "FIREEnergyMinimizerRigidGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#include "SFCPackUpdaterGPU.h"
#include "EAMForceComputeGPU.h"
#include "ConstraintSphereGPU.h"
#include "PotentialPairGPU.h"
#include "PPPMForceComputeGPU.h"
#include "PotentialTersoffGPU.h"
#include "ComputeThermoGPU.h"

#include <cuda_profiler_api.h>
#endif

// include MPI classes
#ifdef ENABLE_MPI
#include "Communicator.h"
#include "DomainDecomposition.h"
#include "LoadBalancer.h"

#ifdef ENABLE_CUDA
#include "CommunicatorGPU.h"
#include "LoadBalancerGPU.h"
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

#include "SignalHandler.h"

#include "HOOMDVersion.h"
#include "PathUtils.h"

#include "num_util.h"

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

using namespace boost::filesystem;
using namespace boost::python;

#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

/* numpy is terrible (see /opt/local/Library/Frameworks/Python.framework/Versions/2.7/
lib/python2.7/site-packages/numpy/core/generate_numpy_array.py)
The following #defines help get around this
*/

#if PY_VERSION_HEX >= 0x03000000
#define MY_PY_VER_3x
#else
#define MY_PY_VER_2x
#endif

#ifdef MY_PY_VER_3x
void *my_import_array()
    {
    import_array();
    return NULL;
    }
#endif
#ifdef MY_PY_VER_2x
void my_import_array()
    {
    import_array();
    }
#endif

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

//! Scans for a VMD installation
/*! \returns Full path to the vmd executable
    This function is for use by the demos and benchmarks installed with HOOMD.
*/
string find_vmd()
    {
    // check some likely locations
    if (exists("/usr/bin/vmd"))
        return "/usr/bin/vmd";
    if (exists("/usr/local/bin/vmd"))
        return "/usr/local/bin/vmd";
    if (exists("/opt/vmd/bin/vmd"))
        return "/opt/vmd/bin/vmd";
    if (exists(path("/Applications/VMD 1.9.app/Contents/Resources/VMD.app/Contents/MacOS/VMD")))
        return("/Applications/VMD 1.9.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");
    if (exists(path("/Applications/VMD 1.8.7.app/Contents/Resources/VMD.app/Contents/MacOS/VMD")))
        return("/Applications/VMD 1.8.7.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");
    if (exists(path("/Applications/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD")))
        return("/Applications/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");

    // return an empty string if we didn't find it
    return "";
    }

//! Method for getting the current version of HOOMD
/*! \returns Current HOOMD version identification string
*/
string get_hoomd_version()
    {
    ostringstream ver;
    // always outputting main version number: #402
    ver << "HOOMD-blue " << HOOMD_VERSION << endl;

    return ver.str();
    }

//! Layer for omp_get_num_procs()
int get_num_procs()
    {
    return 1;
    }

//! Get the hoomd version as a tuple
object get_hoomd_version_tuple()
    {
    return boost::python::make_tuple(HOOMD_VERSION_MAJOR, HOOMD_VERSION_MINOR, HOOMD_VERSION_PATCH);
    }

//! Get the CUDA version as a tuple
object get_cuda_version_tuple()
    {
    #ifdef ENABLE_CUDA
    int major = CUDA_VERSION / 1000;
    int minor = CUDA_VERSION / 10 % 100;
    return boost::python::make_tuple(major, minor);
    #else
    return make_tuple(0,0);
    #endif
    }

//! Get the compiler version
string get_compiler_version()
    {
    #if defined(__GNUC__) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    ostringstream o;
    o << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." <<  __GNUC_PATCHLEVEL__;
    return o.str();

    #elif defined(__clang__)
    ostringstream o;
    o << "clang " << __clang_major__ << "." << __clang_minor__ << "." <<  __clang_patchlevel__;
    return o.str();

    #elif defined(__INTEL_COMPILER)
    ostringstream o;
    o << "icc " << __INTEL_COMPILER;
    return o.str();

    #else
    return string("unknown");

    #endif
    }

//! Determine availability of MPI support
bool is_MPI_available()
   {
   return
#ifdef ENABLE_MPI
       true;
#else
       false;
#endif
    }

void mpi_barrier_world()
    {
    #ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    }

//! Start the CUDA profiler
void cuda_profile_start()
    {
    #ifdef ENABLE_CUDA
    cudaDeviceSynchronize();
    cudaProfilerStart();
    #endif
    }

//! Stop the CUDA profiler
void cuda_profile_stop()
    {
    #ifdef ENABLE_CUDA
    cudaDeviceSynchronize();
    cudaProfilerStop();
    #endif
    }

// values used in measuring hoomd launch timing
unsigned int hoomd_launch_time, hoomd_start_time, hoomd_mpi_init_time;
bool hoomd_launch_timing=false;

#ifdef ENABLE_MPI
//! Environment variables needed for setting up MPI
char env_enable_mpi_cuda[] = "MV2_USE_CUDA=1";

//! Initialize the MPI environment
void initialize_mpi()
    {
    #ifdef ENABLE_MPI_CUDA
    // if we are using an MPI-CUDA implementation, enable this feature
    // before the MPI_Init
    putenv(env_enable_mpi_cuda);
    #endif

    // benchmark hoomd launch times
    if (getenv("HOOMD_LAUNCH_TIME"))
        {
        // get the time that mpirun was called
        hoomd_launch_time = atoi(getenv("HOOMD_LAUNCH_TIME"));

        // compute the number of seconds to get here
        timeval t;
        gettimeofday(&t, NULL);
        hoomd_start_time = t.tv_sec - hoomd_launch_time;
        hoomd_launch_timing = true;
        }

    // initalize MPI
    MPI_Init(0, (char ***) NULL);

    if (hoomd_launch_timing)
        {
        // compute the number of seconds to get past mpi_init
        timeval t;
        gettimeofday(&t, NULL);
        hoomd_mpi_init_time = t.tv_sec - hoomd_launch_time;
        }
    }

//! Get the processor name associated to this rank
string get_mpi_proc_name()
    {
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(proc_name, &name_len);
    return string(proc_name);
    }

//! Finalize MPI environment
void finalize_mpi()
    {
    MPI_Finalize();
    }
#endif

//! Abort MPI runs
void abort_mpi(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    #ifdef ENABLE_MPI
    if(exec_conf->getNRanksGlobal() > 1)
        {
        MPI_Abort(exec_conf->getMPICommunicator(), MPI_ERR_OTHER);
        }
    #endif
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
BOOST_PYTHON_MODULE(hoomd)
    {
    #ifdef ENABLE_MPI
    // initialize MPI early
    initialize_mpi();

    // register clean-up function
    Py_AtExit(finalize_mpi);
    def("get_mpi_proc_name", get_mpi_proc_name);
    #endif

    // setup needed for numpy
    my_import_array();
    boost::python::numpy::initialize();

    def("abort_mpi", abort_mpi);
    def("mpi_barrier_world", mpi_barrier_world);

    def("hoomd_compile_flags", &hoomd_compile_flags);
    def("output_version_info", &output_version_info);
    def("find_vmd", &find_vmd);
    def("get_hoomd_version", &get_hoomd_version);

    def("get_num_procs", &get_num_procs);
    scope().attr("__version__") = get_hoomd_version_tuple();
    scope().attr("__git_sha1__") = HOOMD_GIT_SHA1;
    scope().attr("__git_refspec__") = HOOMD_GIT_REFSPEC;
    scope().attr("__cuda_version__") = get_cuda_version_tuple();
    scope().attr("__compiler_version__") = get_compiler_version();

    def("is_MPI_available", &is_MPI_available);

    def("cuda_profile_start", &cuda_profile_start);
    def("cuda_profile_stop", &cuda_profile_stop);

    // data structures
    class_<std::vector<int> >("std_vector_int")
    .def(vector_indexing_suite<std::vector<int> >());

    class_<std::vector<Scalar3> >("std_vector_scalar3")
    .def(vector_indexing_suite<std::vector<Scalar3> >());

    InstallSIGINTHandler();

    // utils
    export_hoomd_math_functions();
    export_ClockSource();
    export_Profiler();

    // data structures
    export_BoxDim();
    export_ParticleData();
    export_SnapshotParticleData();
    export_RigidData();
    export_SnapshotRigidData();
    export_ExecutionConfiguration();
    export_SystemDefinition();
    export_SnapshotSystemData();
    export_BondedGroupData<BondData,Bond>("BondData","BondDataSnapshot");
    export_BondedGroupData<AngleData,Angle>("AngleData","AngleDataSnapshot");
    export_BondedGroupData<DihedralData,Dihedral>("DihedralData","DihedralDataSnapshot");
    export_BondedGroupData<ImproperData,Dihedral>("ImproperData","ImproperDataSnapshot",false);

    // initializers
    export_RandomInitializer();
    export_SimpleCubicInitializer();
    export_HOOMDInitializer();
    export_RandomGenerator();

    // computes
    export_Compute();
    export_CellList();
    export_CellListStencil();
    export_ForceCompute();
    export_ForceConstraint();
    export_ConstForceCompute();
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
    export_ComputeThermo();
    export_NeighborList();
    export_NeighborListBinned();
    export_NeighborListStencil();
    export_NeighborListTree();
    export_ConstraintSphere();
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
    export_CellListGPU();
    export_NeighborListGPU();
    export_NeighborListGPUBinned();
    export_NeighborListGPUStencil();
    export_NeighborListGPUTree();
    export_CGCMMForceComputeGPU();
    export_PotentialPairGPU<PotentialPairLJGPU, PotentialPairLJ>("PotentialPairLJGPU");
    export_PotentialPairGPU<PotentialPairGaussGPU, PotentialPairGauss>("PotentialPairGaussGPU");
    export_PotentialPairGPU<PotentialPairSLJGPU, PotentialPairSLJ>("PotentialPairSLJGPU");
    export_PotentialPairGPU<PotentialPairYukawaGPU, PotentialPairYukawa>("PotentialPairYukawaGPU");
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
    export_ComputeThermoGPU();
    export_ConstraintSphereGPU();
//    export_ConstExternalFieldDipoleForceComputeGPU();
    export_PPPMForceComputeGPU();
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

    // analyzers
    export_Analyzer();
    export_IMDInterface();
    export_HOOMDDumpWriter();
    export_POSDumpWriter();
    export_PDBDumpWriter();
    export_DCDDumpWriter();
    export_MOL2DumpWriter();
    export_Logger();
    export_MSDAnalyzer();
    export_CallbackAnalyzer();
    export_ParticleGroup();

    // updaters
    export_Updater();
    export_Integrator();
    export_IntegratorTwoStep();
    export_IntegrationMethodTwoStep();
    export_TempRescaleUpdater();
    export_ZeroMomentumUpdater();
    export_SFCPackUpdater();
    export_BoxResizeUpdater();
    export_TwoStepNVE();
    export_TwoStepNVT();
    export_TwoStepNVTMTK();
    export_TwoStepLangevinBase();
    export_TwoStepLangevin();
    export_TwoStepBD();
    export_TwoStepNPTMTK();
    export_Berendsen();
    export_TwoStepNHRigid();
    export_TwoStepNVERigid();
    export_TwoStepNVTRigid();
    export_TwoStepNPHRigid();
    export_TwoStepNPTRigid();
    export_TwoStepBDNVTRigid();
    export_Enforce2DUpdater();
    export_FIREEnergyMinimizer();
    export_FIREEnergyMinimizerRigid();
#ifdef ENABLE_CUDA
    export_SFCPackUpdaterGPU();
    export_TwoStepNVEGPU();
    export_TwoStepNVTGPU();
    export_TwoStepNVTMTKGPU();
    export_TwoStepLangevinGPU();
    export_TwoStepBDGPU();
    export_TwoStepNPTMTKGPU();
    export_BerendsenGPU();
    export_TwoStepNVERigidGPU();
    export_TwoStepNVTRigidGPU();
    export_TwoStepNPHRigidGPU();
    export_TwoStepNPTRigidGPU();
    export_TwoStepBDNVTRigidGPU();
    export_Enforce2DUpdaterGPU();
    export_FIREEnergyMinimizerGPU();
    export_FIREEnergyMinimizerRigidGPU();
#endif

#ifdef ENABLE_MPI
    export_Communicator();
    export_DomainDecomposition();
    export_LoadBalancer();
#ifdef ENABLE_CUDA
    export_CommunicatorGPU();
    export_LoadBalancerGPU();
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

    // system
    export_System();

    // variant
    export_Variant();

    // messenger
    export_Messenger();

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
    register_ptr_to_python< boost::shared_ptr< ComputeThermo > >();
    register_ptr_to_python< boost::shared_ptr< ForceConstraint > >();
    register_ptr_to_python< boost::shared_ptr< BondTablePotential > >();
    register_ptr_to_python< boost::shared_ptr< HarmonicAngleForceCompute > >();
    // register_ptr_to_python< boost::shared_ptr< ForceComputeWrap > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< ExecutionConfiguration > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotRigidData > >();
    register_ptr_to_python< boost::shared_ptr< RigidData > >();
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
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTRigid > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVE > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVT > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPTRigid > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepLangevinBase > >();
    register_ptr_to_python< boost::shared_ptr< Enforce2DUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBD > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTMTK > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPHRigid > >();
    register_ptr_to_python< boost::shared_ptr< BoxResizeUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TempRescaleUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPTMTK > >();
    register_ptr_to_python< boost::shared_ptr< FIREEnergyMinimizerRigid > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBerendsen > >();
    register_ptr_to_python< boost::shared_ptr< IntegratorTwoStep > >();
    // register_ptr_to_python< boost::shared_ptr< UpdaterWrap > >();
    register_ptr_to_python< boost::shared_ptr< Integrator > >();
    register_ptr_to_python< boost::shared_ptr< IntegrationMethodTwoStep > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVERigid > >();
    register_ptr_to_python< boost::shared_ptr< ZeroMomentumUpdater > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepLangevin > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBDNVTRigid > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNHRigid > >();
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
    register_ptr_to_python< boost::shared_ptr< HarmonicDihedralForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CellListGPU > >();
    register_ptr_to_python< boost::shared_ptr< ConstraintSphereGPU > >();
    register_ptr_to_python< boost::shared_ptr< OPLSDihedralForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMAngleForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< ComputeThermoGPU > >();
    register_ptr_to_python< boost::shared_ptr< EAMForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPHRigidGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTRigidGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepLangevinGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVEGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPTMTKGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVTMTKGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBDGPU > >();
    register_ptr_to_python< boost::shared_ptr< FIREEnergyMinimizerGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBerendsenGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepBDNVTRigidGPU > >();
    register_ptr_to_python< boost::shared_ptr< SFCPackUpdaterGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNPTRigidGPU > >();
    register_ptr_to_python< boost::shared_ptr< Enforce2DUpdaterGPU > >();
    register_ptr_to_python< boost::shared_ptr< FIREEnergyMinimizerRigidGPU > >();
    register_ptr_to_python< boost::shared_ptr< TwoStepNVERigidGPU > >();
    #endif

    #ifdef ENABLE_MPI
    register_ptr_to_python< boost::shared_ptr< Communicator > >();
    register_ptr_to_python< boost::shared_ptr< LoadBalancer > >();
    #endif
    #endif
    }
