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

// remove silly warnings
#ifdef WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "ClockSource.h"
#include "Profiler.h"
#include "ParticleData.h"
#include "Initializers.h"
#include "HOOMDInitializer.h"
#include "Compute.h"
#include "ForceCompute.h"
#include "ConstForceCompute.h"
#include "BondForceCompute.h"
#include "LJForceCompute.h"
#include "LJForceComputeThreaded.h"
#include "TempCompute.h"
#include "NeighborList.h"
#include "BinnedNeighborList.h"
#include "Analyzer.h"
#include "IMDInterface.h"
#include "HOOMDDumpWriter.h"
#include "Updater.h"
#include "Integrator.h"
#include "TempRescaleUpdater.h"
#include "SFCPackUpdater.h"
#include "NVTUpdater.h"
#include "NVEUpdater.h"
#include "System.h"

// include GPU classes
#ifdef USE_CUDA
#include "NVTUpdaterGPU.h"
#include "NVEUpdaterGPU.h"
#include "BinnedNeighborListGPU.h"
#include "NeighborListNsqGPU.h"
#include "LJForceComputeGPU.h"
#include "BondForceComputeGPU.h"
#endif

#include "SignalHandler.h"

#include "HOOMDVersion.h"

#include <boost/python.hpp>
#include <boost/filesystem.hpp>

using namespace boost::filesystem;
using namespace boost::python;

#include <iostream>
using namespace std;

/*! \file hoomd_module.cc
	\brief Brings all of the export_* functions together to export the hoomd python module
*/

//! Scans for possible hoomd data directories
/*! \returns the first one it finds
*/
string find_hoomd_data_dir()
	{
	// check likely installation locations
	// default to the source path first, since we would want it to override any
	// other version that exists
	if (exists(path(HOOMD_SOURCE_DIR) / "share"))
		return (path(HOOMD_SOURCE_DIR) / "share").string();
	
	if (getenv("HOOMD_ROOT"))
		{
		path hoomd_root_dir = path(string(getenv("HOOMD_ROOT")));
		if (exists(hoomd_root_dir / "share" / "hoomd"))
			return (hoomd_root_dir / "share" / "hoomd").string();
		}
	
	#ifdef WIN32
	// access the registry key
	string name = string("hoomd ") + string(HOOMD_VERSION);
	string reg_path = "SOFTWARE\\Ames Laboratory Iowa State University\\" + name;
	
	char *value = new char[1024];
	LONG value_size = 1024;
	RegQueryValue(HKEY_LOCAL_MACHINE, reg_path.c_str(), value, &value_size);
	// see if it installed where the reg key says so
	path install_dir = path(string(value));
	if (exists(install_dir))
		return (install_dir).string();
	delete[] value;

	// otherwise, check the program files root
	if (getenv("PROGRAMFILES"))
		{
		path program_files_dir = path(string(getenv("PROGRAMFILES")));
		if (exists(program_files_dir / "hoomd"))
			return (program_files_dir / "hoomd").string();
		}
	#else
	if (exists("/usr/share/hoomd"))
		return "/usr/share/hoomd";
	if (exists("/usr/local/share/hoomd"))
		return "/usr/local/share/hoomd";
	#endif
	
	cerr << "HOOMD data directory not found, please set the environment variable HOOMD_ROOT" << endl;
	return string("");
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

	InstallSIGINTHandler();

	// utils
	export_ClockSource();
	export_Profiler();

	// data structures
	export_BoxDim();
	export_ParticleDataInitializer();
	export_ParticleData();

	// initializers
	export_RandomInitializer();
	export_SimpleCubicInitializer();
	export_HOOMDInitializer();
	
	// computes
	export_Compute();
	export_ForceCompute();
	export_ConstForceCompute();
	export_BondForceCompute();
	export_LJForceCompute();
	export_LJForceComputeThreaded();
	export_TempCompute();
	export_NeighborList();
	export_BinnedNeighborList();
	#ifdef USE_CUDA
	export_BinnedNeighborListGPU();
	export_NeighborListNsqGPU();
	export_LJForceComputeGPU();
	export_BondForceComputeGPU();
	#endif
	
	// analyzers
	export_Analyzer();
	export_IMDInterface();
	export_HOOMDDumpWriter();
	
	// updaters
	export_Updater();
	export_Integrator();
	export_TempRescaleUpdater();
	export_SFCPackUpdater();
	export_NVTUpdater();
	export_NVEUpdater();
	#ifdef USE_CUDA
	export_NVEUpdaterGPU();
	export_NVTUpdaterGPU();
	#endif

	// system
	export_System();
	
	}
