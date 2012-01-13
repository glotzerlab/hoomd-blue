/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

//! Enable old boost::filesystem API (temporary fix)
#define BOOST_FILESYSTEM_VERSION 2

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
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"
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
#include "HarmonicBondForceCompute.h"
#include "HarmonicAngleForceCompute.h"
#include "HarmonicDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"
#include "CGCMMAngleForceCompute.h"
#include "FENEBondForceCompute.h"
#include "CGCMMForceCompute.h"
#include "TablePotential.h"
#include "LJWallForceCompute.h"
#include "AllPairPotentials.h"
#include "ComputeThermo.h"
#include "ComputeThermoGPU.h"
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
#include "TwoStepNPT.h"
#include "TwoStepBerendsen.h"
#include "TwoStepNVERigid.h" 
#include "TwoStepNVTRigid.h"
#include "TwoStepNPTRigid.h"  
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
#include "EvaluatorPairDPDThermo.h"
#include "PotentialPairDPDLJThermo.h"
#include "EvaluatorPairDPDLJThermo.h"
#include "PotentialPair.h"
#include "PPPMForceCompute.h"


// include GPU classes
#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#include "TwoStepNVEGPU.h"
#include "TwoStepNVTGPU.h"
#include "TwoStepBDNVTGPU.h"
#include "TwoStepNPTGPU.h"
#include "TwoStepBerendsenGPU.h"
#include "TwoStepNVERigidGPU.h" 
#include "TwoStepNVTRigidGPU.h" 
#include "TwoStepNPTRigidGPU.h" 
#include "TwoStepBDNVTRigidGPU.h" 
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
#include "CGCMMForceComputeGPU.h"
//#include "ConstExternalFieldDipoleForceComputeGPU.h"
#include "TablePotentialGPU.h"
#include "HarmonicBondForceComputeGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#include "HarmonicDihedralForceComputeGPU.h"
#include "HarmonicImproperForceComputeGPU.h"
#include "CGCMMAngleForceComputeGPU.h"
#include "FENEBondForceComputeGPU.h"
#include "Enforce2DUpdaterGPU.h"
#include "FIREEnergyMinimizerRigidGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#include "EAMForceComputeGPU.h"
#include "ConstraintSphereGPU.h"
#include "PotentialPairGPU.h"
#include "PPPMForceComputeGPU.h"
#endif

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

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

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
    if (exists(path("/Applications/VMD 1.9.app/Contents/Resources/VMD.app/Contents/MacOS/VMD", no_check)))
        return("/Applications/VMD 1.9.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");
    if (exists(path("/Applications/VMD 1.8.7.app/Contents/Resources/VMD.app/Contents/MacOS/VMD", no_check)))
        return("/Applications/VMD 1.8.7.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");
    if (exists(path("/Applications/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD", no_check)))
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

//! Layer for omp_set_num_threads
void set_num_threads(int nthreads)
    {
    #ifdef ENABLE_OPENMP
    omp_set_num_threads(nthreads);
    #endif
    }

//! Layer for omp_get_num_procs()
int get_num_procs()
    {
    #ifdef ENABLE_OPENMP
    return omp_get_num_procs();
    #else
    return 1;
    #endif
    }

//! Get the hoomd version as a tuple
object get_version_tuple()
    {
    return make_tuple(HOOMD_VERSION_MAJOR, HOOMD_VERSION_MINOR, HOOMD_VERSION_PATCH);
    }

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(hoomd)
    {
    // write out the version information on the module import
    output_version_info(false);
    def("find_vmd", &find_vmd);
    def("get_hoomd_version", &get_hoomd_version);

    def("set_num_threads", &set_num_threads);
    def("get_num_procs", &get_num_procs);
    scope().attr("__version__") = get_version_tuple();

    // data structures
    class_<std::vector<int> >("std_vector_int")
    .def(vector_indexing_suite<std::vector<int> >());
    
    InstallSIGINTHandler();
    
    // utils
    export_hoomd_math_functions();
    export_ClockSource();
    export_Profiler();
    
    // data structures
    export_BoxDim();
    export_ParticleDataInitializer();
    export_ParticleData();
    export_RigidData();
    export_ExecutionConfiguration();
    export_BondData();
    export_SystemDefinition();
    export_AngleData();
    export_DihedralData();
    
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
    export_HarmonicBondForceCompute();
    export_HarmonicAngleForceCompute();
    export_HarmonicDihedralForceCompute();
    export_HarmonicImproperForceCompute();
    export_CGCMMAngleForceCompute();
    export_TablePotential();
    export_FENEBondForceCompute();
    export_CGCMMForceCompute();
    export_PotentialPair<PotentialPairLJ>("PotentialPairLJ");
    export_PotentialPair<PotentialPairGauss>("PotentialPairGauss");
    export_PotentialPair<PotentialPairSLJ>("PotentialPairSLJ");
    export_PotentialPair<PotentialPairYukawa>("PotentialPairYukawa");
    export_PotentialPair<PotentialPairEwald>("PotentialPairEwald");
    export_PotentialPair<PotentialPairMorse>("PotentialPairMorse");
    export_PotentialPair<PotentialPairDPD> ("PotentialPairDPD");
    export_PotentialPairDPDThermo<PotentialPairDPDThermoDPD, PotentialPairDPD>("PotentialPairDPDThermoDPD");   
    export_PotentialPair<PotentialPairDPDLJ> ("PotentialPairDPDLJ");
    export_PotentialPairDPDLJThermo<PotentialPairDPDLJThermoDPD, PotentialPairDPDLJ>("PotentialPairDPDLJThermoDPD");
    export_EAMForceCompute();
    export_LJWallForceCompute();
    export_ComputeThermo();
    export_NeighborList();
    export_NeighborListBinned();
    export_ConstraintSphere();
    export_PPPMForceCompute();
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
    export_PotentialPairDPDThermoGPU<PotentialPairDPDThermoDPDGPU, PotentialPairDPDThermoDPD >("PotentialPairDPDThermoDPDGPU");    
    export_PotentialPairGPU<PotentialPairDPDLJGPU, PotentialPairDPDLJ> ("PotentialPairDPDLJGPU");    
    export_PotentialPairDPDLJThermoGPU<PotentialPairDPDLJThermoDPDGPU, PotentialPairDPDLJThermoDPD >("PotentialPairDPDLJThermoDPDGPU");    
    export_TablePotentialGPU();
    export_EAMForceComputeGPU();
    export_HarmonicBondForceComputeGPU();
    export_HarmonicAngleForceComputeGPU();
    export_HarmonicDihedralForceComputeGPU();
    export_HarmonicImproperForceComputeGPU();
    export_CGCMMAngleForceComputeGPU();
    export_FENEBondForceComputeGPU();
    export_ComputeThermoGPU();
    export_ConstraintSphereGPU();
//    export_ConstExternalFieldDipoleForceComputeGPU();
    export_PPPMForceComputeGPU();
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
    export_TwoStepNPT();
    export_Berendsen();
    export_TwoStepNVERigid();
    export_TwoStepNVTRigid();
    export_TwoStepNPTRigid();
    export_TwoStepBDNVTRigid();
    export_Enforce2DUpdater();
    export_FIREEnergyMinimizer();
    export_FIREEnergyMinimizerRigid();        
#ifdef ENABLE_CUDA
    export_TwoStepNVEGPU();
    export_TwoStepNVTGPU();
    export_TwoStepBDNVTGPU();
    export_TwoStepNPTGPU();
    export_BerendsenGPU();
    export_TwoStepNVERigidGPU();
    export_TwoStepNVTRigidGPU();
    export_TwoStepNPTRigidGPU();
    export_TwoStepBDNVTRigidGPU();
    export_Enforce2DUpdaterGPU();
    export_FIREEnergyMinimizerGPU();
    export_FIREEnergyMinimizerRigidGPU();          
#endif
    
    // system
    export_System();
    
    // variant
    export_Variant();
    
    }

#ifdef WIN32
#pragma warning( pop )
#endif

