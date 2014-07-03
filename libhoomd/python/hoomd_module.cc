/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 4267 )
#endif

#include "HOOMDMath.h"
#include "ClockSource.h"
#include "Profiler.h"
#include "ParticleData.h"
#include "RigidData.h"
#include "SystemDefinition.h"
#include "BondedGroupData.h"
#include "ExecutionConfiguration.h"
#include "Initializers.h"
#include "HOOMDInitializer.h"
#include "HOOMDBinaryInitializer.h"
#include "RandomGenerator.h"
#include "Compute.h"
#include "CellList.h"
#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "ConstForceCompute.h"
#include "ConstExternalFieldDipoleForceCompute.h"
#include "HarmonicAngleForceCompute.h"
#include "TableAngleForceCompute.h"
#include "HarmonicDihedralForceCompute.h"
#include "TableDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"
#include "CGCMMAngleForceCompute.h"
#include "CGCMMForceCompute.h"
#include "TablePotential.h"
#include "BondTablePotential.h"
#include "LJWallForceCompute.h"
#include "AllPairPotentials.h"
#include "AllBondPotentials.h"
#include "AllTripletPotentials.h"
#include "ComputeThermo.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "Analyzer.h"
#include "IMDInterface.h"
#include "HOOMDDumpWriter.h"
#include "HOOMDBinaryDumpWriter.h"
#include "PDBDumpWriter.h"
#include "MOL2DumpWriter.h"
#include "DCDDumpWriter.h"
#include "Logger.h"
#include "MSDAnalyzer.h"
#include "Updater.h"
#include "Integrator.h"
#include "IntegratorTwoStep.h"
#include "IntegrationMethodTwoStep.h"
#include "TwoStepNVE.h"
#include "TwoStepNVT.h"
#include "TwoStepBDNVT.h"
#include "TwoStepNPTMTK.h"
#include "TwoStepBerendsen.h"
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
#include "TwoStepBDNVTGPU.h"
#include "TwoStepNPTMTKGPU.h"
#include "TwoStepBerendsenGPU.h"
#include "TwoStepNVERigidGPU.h"
#include "TwoStepNVTRigidGPU.h"
#include "TwoStepNPHRigidGPU.h"
#include "TwoStepNPTRigidGPU.h"
#include "TwoStepBDNVTRigidGPU.h"
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
#include "CGCMMForceComputeGPU.h"
//#include "ConstExternalFieldDipoleForceComputeGPU.h"
#include "BondTablePotentialGPU.h"
#include "TablePotentialGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#include "TableAngleForceComputeGPU.h"
#include "HarmonicDihedralForceComputeGPU.h"
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

#ifdef ENABLE_CUDA
#include "CommunicatorGPU.h"
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

#include "SignalHandler.h"

#include "HOOMDVersion.h"
#include "PathUtils.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

using namespace boost::filesystem;
using namespace boost::python;

#include <iostream>
#include <sstream>
using namespace std;

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

//! Scans for a VMD installation
/*! \returns Full path to the vmd executable
    This function is for use by the demos and benchmarks installed with HOOMD.
*/
string find_vmd()
    {
#ifdef WIN32

    // find VMD through the registry
    vector<string> reg_paths;
    reg_paths.push_back("SOFTWARE\\University of Illinois\\VMD\\1.9.1");
    reg_paths.push_back("SOFTWARE\\University of Illinois\\VMD\\1.9.0");
    reg_paths.push_back("SOFTWARE\\University of Illinois\\VMD\\1.9");
    reg_paths.push_back("SOFTWARE\\University of Illinois\\VMD\\1.8.7");
    reg_paths.push_back("SOFTWARE\\University of Illinois\\VMD\\1.8.6");

    vector<string>::iterator cur_path;
    for (cur_path = reg_paths.begin(); cur_path != reg_paths.end(); ++cur_path)
        {
        string reg_path = *cur_path;

        char *value = new char[1024];
        DWORD value_size = 1024;
        HKEY vmd_root_key;
        LONG err_code = RegOpenKeyEx(HKEY_LOCAL_MACHINE, reg_path.c_str(), 0, KEY_READ | KEY_WOW64_32KEY, &vmd_root_key);
        if (err_code == ERROR_SUCCESS)
            {
            err_code = RegQueryValueEx(vmd_root_key, "VMDDIR", NULL, NULL, (LPBYTE)value, &value_size);
            // see if it installed where the reg key says so
            if (err_code == ERROR_SUCCESS)
                {
                path install_dir = path(string(value));
                if (exists(install_dir / "vmd.exe"))
                    return (install_dir / "vmd.exe").string();
                }
            }

        err_code = RegOpenKeyEx(HKEY_LOCAL_MACHINE, reg_path.c_str(), 0, KEY_READ, &vmd_root_key);
        if (err_code == ERROR_SUCCESS)
            {
            err_code = RegQueryValueEx(vmd_root_key, "VMDDIR", NULL, NULL, (LPBYTE)value, &value_size);
            // see if it installed where the reg key says so
            if (err_code == ERROR_SUCCESS)
                {
                path install_dir = path(string(value));
                if (exists(install_dir / "vmd.exe"))
                    return (install_dir / "vmd.exe").string();
                }
            }

        delete[] value;
        }

#else
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
#endif

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
    return make_tuple(HOOMD_VERSION_MAJOR, HOOMD_VERSION_MINOR, HOOMD_VERSION_PATCH);
    }

//! Get the CUDA version as a tuple
object get_cuda_version_tuple()
    {
    #ifdef ENABLE_CUDA
    int major = CUDA_VERSION / 1000;
    int minor = CUDA_VERSION / 10 % 100;
    return make_tuple(major, minor);
    #else
    return make_tuple(0,0);
    #endif
    }

//! Get the compiler version
string get_compiler_version()
    {
    #ifdef __GNUC__
    ostringstream o;
    o << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." <<  __GNUC_PATCHLEVEL__;
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

    // initalize MPI
    MPI_Init(0, (char ***) NULL);
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
    if (exec_conf->getNRanksGlobal() > 1)
        {
        MPI_Abort(exec_conf->getMPICommunicator(), MPI_ERR_OTHER);
        }
    #endif
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
    #endif

    def("abort_mpi", abort_mpi);

    // write out the version information on the module import
    output_version_info(false);
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
    export_RandomInitializerWithWalls();
    export_SimpleCubicInitializer();
    export_HOOMDInitializer();
    export_HOOMDBinaryInitializer();
    export_RandomGenerator();

    // computes
    export_Compute();
    export_CellList();
    export_ForceCompute();
    export_ForceConstraint();
    export_ConstForceCompute();
    export_ConstExternalFieldDipoleForceCompute();
    export_HarmonicAngleForceCompute();
    export_TableAngleForceCompute();
    export_HarmonicDihedralForceCompute();
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
    export_tersoff_params();
    export_PotentialPair<PotentialPairForceShiftedLJ>("PotentialPairForceShiftedLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDThermoDPD, PotentialPairDPD>("PotentialPairDPDThermoDPD");
    export_PotentialPair<PotentialPairDPDLJ> ("PotentialPairDPDLJ");
    export_PotentialPairDPDThermo<PotentialPairDPDLJThermoDPD, PotentialPairDPDLJ>("PotentialPairDPDLJThermoDPD");
    export_PotentialBond<PotentialBondHarmonic>("PotentialBondHarmonic");
    export_PotentialBond<PotentialBondFENE>("PotentialBondFENE");
    export_EAMForceCompute();
    export_LJWallForceCompute();
    export_ComputeThermo();
    export_NeighborList();
    export_NeighborListBinned();
    export_ConstraintSphere();
    export_PPPMForceCompute();
    export_PotentialExternal<PotentialExternalPeriodic>("PotentialExternalPeriodic");
#ifdef ENABLE_CUDA
    export_CellListGPU();
    export_NeighborListGPU();
    export_NeighborListGPUBinned();
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
    export_PotentialPairDPDThermoGPU<PotentialPairDPDThermoDPDGPU, PotentialPairDPDThermoDPD >("PotentialPairDPDThermoDPDGPU");
    export_PotentialPairGPU<PotentialPairDPDLJGPU, PotentialPairDPDLJ> ("PotentialPairDPDLJGPU");
    export_PotentialPairDPDThermoGPU<PotentialPairDPDLJThermoDPDGPU, PotentialPairDPDLJThermoDPD >("PotentialPairDPDLJThermoDPDGPU");
    export_PotentialBondGPU<PotentialBondHarmonicGPU, PotentialBondHarmonic>("PotentialBondHarmonicGPU");
    export_PotentialBondGPU<PotentialBondFENEGPU, PotentialBondFENE>("PotentialBondFENEGPU");
    export_BondTablePotentialGPU();
    export_TablePotentialGPU();
    export_EAMForceComputeGPU();
    export_HarmonicAngleForceComputeGPU();
    export_TableAngleForceComputeGPU();
    export_HarmonicDihedralForceComputeGPU();
    export_TableDihedralForceComputeGPU();
    export_HarmonicImproperForceComputeGPU();
    export_CGCMMAngleForceComputeGPU();
    export_ComputeThermoGPU();
    export_ConstraintSphereGPU();
//    export_ConstExternalFieldDipoleForceComputeGPU();
    export_PPPMForceComputeGPU();
    export_PotentialExternalGPU<PotentialExternalPeriodicGPU, PotentialExternalPeriodic>("PotentialExternalPeriodicGPU");
#endif

    // analyzers
    export_Analyzer();
    export_IMDInterface();
    export_HOOMDDumpWriter();
    export_HOOMDBinaryDumpWriter();
    export_PDBDumpWriter();
    export_DCDDumpWriter();
    export_MOL2DumpWriter();
    export_Logger();
    export_MSDAnalyzer();
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
    export_TwoStepBDNVT();
    export_TwoStepNPTMTK();
    export_Berendsen();
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
    export_TwoStepBDNVTGPU();
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
#ifdef ENABLE_CUDA
    export_CommunicatorGPU();
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

    // system
    export_System();

    // variant
    export_Variant();

    // messenger
    export_Messenger();
    }

#ifdef WIN32
#pragma warning( pop )
#endif

