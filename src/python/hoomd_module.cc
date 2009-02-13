/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

// remove silly warnings
#ifdef WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "ClockSource.h"
#include "Profiler.h"
#include "ParticleData.h"
#include "SystemDefinition.h"
#include "BondData.h"
#include "ExecutionConfiguration.h"
#include "Initializers.h"
#include "HOOMDInitializer.h"
#include "RandomGenerator.h"
#include "Compute.h"
#include "ForceCompute.h"
#include "ConstForceCompute.h"
#include "HarmonicBondForceCompute.h"
#include "FENEBondForceCompute.h"
#include "LJForceCompute.h"
#include "YukawaForceCompute.h"
#include "LJWallForceCompute.h"
#include "TempCompute.h"
#include "NeighborList.h"
#include "BinnedNeighborList.h"
#include "Analyzer.h"
#include "IMDInterface.h"
#include "HOOMDDumpWriter.h"
#include "PDBDumpWriter.h"
#include "MOL2DumpWriter.h"
#include "DCDDumpWriter.h"
#include "Logger.h"
#include "MSDAnalyzer.h"
#include "Updater.h"
#include "Integrator.h"
#include "TempRescaleUpdater.h"
#include "ZeroMomentumUpdater.h"
#include "SFCPackUpdater.h"
#include "NVTUpdater.h"
#include "NPTUpdater.h"
#include "NVEUpdater.h"
#include "BD_NVTUpdater.h"
#include "System.h"

// include GPU classes
#ifdef ENABLE_CUDA
#include "NVTUpdaterGPU.h"
#include "NVEUpdaterGPU.h"
#include "NPTUpdaterGPU.h"
#include "BD_NVTUpdaterGPU.h"
#include "BinnedNeighborListGPU.h"
#include "NeighborListNsqGPU.h"
#include "LJForceComputeGPU.h"
#include "YukawaForceComputeGPU.h"
#include "StochasticForceComputeGPU.h"
#include "HarmonicBondForceComputeGPU.h"
#include "FENEBondForceComputeGPU.h"
#endif

#include "SignalHandler.h"

#include "HOOMDVersion.h"

#include <boost/python.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

using namespace boost::filesystem;
using namespace boost::python;

#include <iostream>
using namespace std;

// include gpu_settings.h for g_gpu_error_checking
#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#else
// otherwise, we need a dummy variable that will do nothing
bool g_gpu_error_checking = false;
#endif

/*! \file hoomd_module.cc
	\brief Brings all of the export_* functions together to export the hoomd python module
*/

//! Scans for possible hoomd data directories
/*! \returns the first one it finds
	This function is for use by the demos and benchmarks installed with HOOMD.
*/
string find_hoomd_data_dir()
	{
	// try checking offsets from the environment variable first:
	// it is searched first so as to override any other potential location
	if (getenv("HOOMD_ROOT"))
		{
		path hoomd_root_dir = path(string(getenv("HOOMD_ROOT")));
		// try root/share/hoomd (for /usr/local /usr etc)
		if (exists(hoomd_root_dir / "share" / "hoomd" / "hoomd_data_dir"))
			{
			string result = (hoomd_root_dir / "share" / "hoomd").string();
			return result;
			}
		// try root/share (for /opt/hoomd style install)
		if (exists(hoomd_root_dir / "share" / "hoomd_data_dir"))
			{
			string result = (hoomd_root_dir / "share").string();
			return result;
			}
		}

	// as a final fallback: try the source directory
	if (exists(path(HOOMD_SOURCE_DIR) / "share" / "hoomd" / "hoomd_data_dir"))
		return (path(HOOMD_SOURCE_DIR) / "share" / "hoomd").string();

	#ifdef WIN32
	// access the registry key
	string name = string("hoomd ") + string(HOOMD_VERSION);
	string reg_path = "SOFTWARE\\Ames Laboratory Iowa State University\\" + name;
	
	char *value = new char[1024];
	LONG value_size = 1024;
	LONG err_code = RegQueryValue(HKEY_LOCAL_MACHINE, reg_path.c_str(), value, &value_size);
	// see if it installed where the reg key says so
	if (err_code == ERROR_SUCCESS)
		{
		path install_dir = path(string(value));
		if (exists(install_dir / "hoomd_data_dir"))
			return (install_dir).string();
		}
	delete[] value;


	// otherwise, check the program files root
	if (getenv("PROGRAMFILES"))
		{
		path program_files_dir = path(string(getenv("PROGRAMFILES")));
		if (exists(program_files_dir / "hoomd" / "hoomd_data_dir"))
			return (program_files_dir / "hoomd").string();
		}
	#else
	// check a few likely installation locations
	if (exists("/usr/share/hoomd/hoomd_data_dir"))
		return "/usr/share/hoomd";
	if (exists("/usr/local/share/hoomd/hoomd_data_dir"))
		return "/usr/local/share/hoomd";
	if (exists("/opt/hoomd/share/hoomd_data_dir"))
		return "/opt/hoomd/share";
	#endif

	cerr << endl << "***Error! HOOMD data directory not found, please set the environment variable HOOMD_ROOT" << endl << endl;
	return string("");
	}
	
//! Scans for a VMD installation
/*! \returns Full path to the vmd executable
	This function is for use by the demos and benchmarks installed with HOOMD.
*/
string find_vmd()
	{
	#ifdef WIN32
	
	// find VMD through the registry
	string reg_path = "SOFTWARE\\University of Illinois\\VMD\\1.8.6";
	
	char *value = new char[1024];
	DWORD value_size = 1024;
	HKEY vmd_root_key;
	LONG err_code = RegOpenKeyEx(HKEY_LOCAL_MACHINE, reg_path.c_str(), 0, KEY_READ, &vmd_root_key);
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

	#else
	// check some likely locations
	if (exists("/usr/bin/vmd"))
		return "/usr/bin/vmd";
	if (exists("/usr/local/bin/vmd"))
		return "/usr/local/bin/vmd";
	if (exists("/opt/vmd/bin/vmd"))
		return "/opt/vmd/bin/vmd";
	if (exists(path("/Applications/3rd Party Apps/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD", no_check )))
		return("/Applications/3rd Party Apps/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");	
	if (exists(path("/Applications/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD", no_check)))
		return("/Applications/VMD 1.8.6.app/Contents/Resources/VMD.app/Contents/MacOS/VMD");
	#endif
	
	// return an empty string if we didn't find it
	return "";
	}

//! Method for setting g_gpu_error_checking
/*! \param value Value to set
	Sets the value of g_gpu_error_checking to enable or disable error checking on the GPU
*/
void set_gpu_error_checking(bool value)
	{
	g_gpu_error_checking = value;
	}

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
	create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(hoomd)
	{
	// write out the version information on the module import
	output_version_info(false);
	def("find_hoomd_data_dir", &find_hoomd_data_dir);
	def("find_vmd", &find_vmd);
	def("set_gpu_error_checking", &set_gpu_error_checking);

	InstallSIGINTHandler();

	// utils
	export_ClockSource();
	export_Profiler();

	// data structures
	export_BoxDim();
	export_ParticleDataInitializer();
	export_ParticleData();
	export_ExecutionConfiguration();
	export_BondData();
	export_SystemDefinition();

	// initializers
	export_RandomInitializer();
	export_RandomInitializerWithWalls();
	export_SimpleCubicInitializer();
	export_HOOMDInitializer();
	export_RandomGenerator();
	
	// computes
	export_Compute();
	export_ForceCompute();
	export_ConstForceCompute();
	export_HarmonicBondForceCompute();
	export_FENEBondForceCompute();
	export_LJForceCompute();
	export_YukawaForceCompute();	
	export_LJWallForceCompute();
	export_TempCompute();
	export_NeighborList();
	export_BinnedNeighborList();
	#ifdef ENABLE_CUDA
	export_BinnedNeighborListGPU();
	export_NeighborListNsqGPU();
	export_LJForceComputeGPU();
	export_YukawaForceComputeGPU();
	export_HarmonicBondForceComputeGPU();
	export_FENEBondForceComputeGPU();
	#endif
	
	// analyzers
	export_Analyzer();
	export_IMDInterface();
	export_HOOMDDumpWriter();
	export_PDBDumpWriter();
	export_DCDDumpWriter();
	export_MOL2DumpWriter();
	export_Logger();
	export_MSDAnalyzer();
	export_ParticleGroup();
	
	// updaters
	export_Updater();
	export_Integrator();
	export_TempRescaleUpdater();
	export_ZeroMomentumUpdater();
	export_SFCPackUpdater();
	export_NVTUpdater();
	export_NPTUpdater();
	export_NVEUpdater();
	export_BD_NVTUpdater();
	#ifdef ENABLE_CUDA
	export_NVEUpdaterGPU();
	export_NVTUpdaterGPU();
	export_NPTUpdaterGPU();
	export_BD_NVTUpdaterGPU();
	#endif

	// system
	export_System();
	
	}

#ifdef WIN32
#pragma warning( pop )
#endif

