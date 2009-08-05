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
// Maintainer: joaander

#include "ExecutionConfiguration.h"

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "gpu_settings.h"
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace boost;

/*! \file ExecutionConfiguration.cc
    \brief Defines ExecutionConfiguration and related classes
*/

/*! \param min_cpu If set to true, cudaDeviceBlockingSync is passed to GPUWorker to keep the CPU usage of HOOMD to a
                   minimum
    \param ignore_display If set to true, try to ignore GPUs attached to the display

    If there are capable GPUs present in the system, the default chosen by CUDA will be used. Specifically, 
    cudaSetDevice is not called, so systems with compute-exclusive GPUs will see automatic choice of free GPUs.
    If there are no capable GPUs present in the system, then the execution mode will revert run on the CPU.
*/
ExecutionConfiguration::ExecutionConfiguration(bool min_cpu, bool ignore_display)
	{
	#ifdef ENABLE_CUDA
	// scan the available GPUs
	scanGPUs(ignore_display);
	
	// if there are available GPUs, initialize them. Otherwise, default to running on the CPU	
	int dev_count = getNumCapableGPUs();
	std::vector<int> gpu_ids;
		
	if (dev_count > 0)
		{
		#ifdef ENABLE_CAC_GPU_ID
		gpu_ids = getDefaultGPUList();
		#else
		gpu_ids.push_back(-1);
		#endif
		exec_mode = GPU;
		
		initializeGPUs(gpu_ids, min_cpu);
		}
	else
		exec_mode = CPU;

	#else
	exec_mode=CPU;
	#endif
	}

/*! \param mode Execution mode to set (cpu or gpu)
    \param min_cpu If set to true, cudaDeviceBlockingSync is passed to GPUWorker to keep the CPU usage of HOOMD to a
                   minimum
    \param ignore_display If set to true, try to ignore GPUs attached to the display

    Explicitly force the use of either CPU or GPU execution. If GPU exeuction is selected, then a default GPU choice
    is made by not calling cudaSetDevice.
*/
ExecutionConfiguration::ExecutionConfiguration(executionMode mode, bool min_cpu, bool ignore_display)
	{
	exec_mode = mode;
	
	#ifdef ENABLE_CUDA
	// scan the available GPUs
	scanGPUs(ignore_display);
	
	// initialize the GPU if that mode was requested
	if (exec_mode == GPU)
		{
		std::vector<int> gpu_ids;
		#ifdef ENABLE_CAC_GPU_ID
		gpu_ids = getDefaultGPUList();
		#else
		gpu_ids.push_back(-1);
		#endif
		exec_mode = GPU;
			
		initializeGPUs(gpu_ids, min_cpu);
		}
	#endif
	}
	
/*! \param gpu_ids List of GPUs to execute on
    \param min_cpu If set to true, cudaDeviceBlockingSync is passed to GPUWorker to keep the CPU usage of HOOMD to a
    minimum
    \param ignore_display If set to true, try to ignore GPUs attached to the display

    Run only on the specified GPUs
*/
ExecutionConfiguration::ExecutionConfiguration(const std::vector<int>& gpu_ids, bool min_cpu, bool ignore_display)
	{
	exec_mode = GPU;

	#ifdef ENABLE_CAC_GPU_ID
	cerr << endl << "***Error! Do not specify --mode=gpu when running on CAC, it triggers a nasty bug" << endl;
	throw runtime_error("Error initializing execution configuration");
	#endif

	#ifdef ENABLE_CUDA
	// scan the available GPUs
	scanGPUs(ignore_display);
	initializeGPUs(gpu_ids, min_cpu);
	#else
	cout << endl << "***Error! GPU execution was requested, but this build of HOOMD is for CPUs only" << endl
			<< endl;
	throw runtime_error("Error initializing execution configuration");
	#endif
	}
	
#ifdef ENABLE_CUDA
/*! \param file Passed to GPUWorker::setTag
	\param line Passed to GPUWorker::setTag
*/
void ExecutionConfiguration::tagAll(const std::string &file, unsigned int line) const
	{
	for (unsigned int i = 0; i < gpu.size(); i++)
		gpu[i]->setTag(file, line);
	}
	
/*! Calls GPUWorker::sync() for all GPUs in the configuration
*/
void ExecutionConfiguration::syncAll() const
	{
	for (unsigned int i = 0; i < gpu.size(); i++)
		gpu[i]->sync();
	}
	
/*! \param func Passed to GPUWorker::call
*/
void ExecutionConfiguration::callAll(const boost::function< cudaError_t (void) > &func) const
	{
	for (unsigned int i = 0; i < gpu.size(); i++)
		gpu[i]->call(func);
	}

/*! \returns 0 in normal builds
	\returns the value of the environment variable $CAC_GPU_ID when built with \a ENABLE_CAC_GPU_ID turned on in CMake
*/
int ExecutionConfiguration::getDefaultGPU()
	{
	#ifdef ENABLE_CAC_GPU_ID
	char *id_str = getenv("CAC_GPU_ID");
	if (id_str)
		{
		unsigned int id = atoi(id_str);
		cout << "Notice: HOOMD is running on GPU " << id << " as specified by $CAC_GPU_ID" << endl;
		return id;
		}
	else
		{
		cerr << endl << "***Warning! HOOMD built with CAC_GPU_ID support, but no CAC_GPU_ID specified!" << endl << endl;
		throw runtime_error("Error initializing execution configuration");
		}
	#else
	return 0;
	#endif
	}

#ifdef ENABLE_CAC_GPU_ID
//! simple tokenizer
/*! \param str String to tokenize
	\param delimiters Delmiters that break up the string
	This code originated here: http://www.digitalpeer.com/id/simple
*/
static vector<string> tokenize(const string& str,const string& delimiters)
	{
	vector<string> tokens;
	
	// skip delimiters at beginning.
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
		
	// find first "non-delimiter".
	string::size_type pos = str.find_first_of(delimiters, lastPos);
	
	while (string::npos != pos || string::npos != lastPos)	
		{
		// found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
			
		// skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		
		// find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
		}
	
	return tokens;
	}
#endif
	
/*! \returns a list with 0 in it in normal builds
	\returns the value of the environment variable $CAC_GPU_ID when built with \a ENABLE_CAC_GPU_ID turned on in CMake
*/
std::vector< int > ExecutionConfiguration::getDefaultGPUList()
	{
	#ifdef ENABLE_CAC_GPU_ID
	char *id_str = getenv("CAC_GPU_ID");
	if (id_str)
		{
		vector<string> tokens = tokenize(string(id_str), ",");
		vector< int > result;
		for (unsigned int i = 0; i < tokens.size(); i++)
			{
			int id = atoi(tokens[i].c_str());
			if (id < 0)
				{
				cout << endl << "***Error! CAC_GPU_ID contains negative values!" << endl << endl;
				throw runtime_error("Error initializing execution configuration");
				}
				
			result.push_back(id);
			}
		
		cout << "Notice: HOOMD is running on ";
		if (result.size() == 1)
			cout << "GPU ";
		else
			cout << "the GPUs ";

		for (unsigned int i = 0; i < result.size(); i++)
			{
			cout << result[i];
			if (i != result.size()-1)
				cout << ",";
			}
		
		cout << " as specified by $CAC_GPU_ID" << endl;
		
		return result;
		}
	else
		{
		cerr << endl << "***Warning! HOOMD built with CAC_GPU_ID support, but no CAC_GPU_ID specified!" << endl << endl;
		throw runtime_error("Error initializing execution configuration");
		}
	#else
	vector< int > result;
	result.push_back(0);
	return result;
	#endif
	}

/*! \returns Compute capability of GPU 0 as a string
    \note Silently returns an emtpy string if no GPUs are specified
*/
std::string ExecutionConfiguration::getComputeCapability()
	{
	ostringstream s;
	
	if (gpu.size() > 0)
		{
		cudaDeviceProp dev_prop;
		int dev;
			
		// get the device and poperties
		gpu[0]->call(bind(cudaGetDevice, &dev));
		gpu[0]->call(bind(cudaGetDeviceProperties, &dev_prop, dev));
		
		s << dev_prop.major << "." << dev_prop.minor;
		}
	
	return s.str();
	}


/*! \param gpu_ids List of GPU ids to initialize
    \param min_cpu If set to true, cudaDeviceBlockingSync is passed to GPUWorker to keep the CPU usage of HOOMD to a
                   minimum

    \pre scanGPUs has been called

    initializeGPUs will loop through the specified list of GPUs, validate that each one is available for CUDA use
    and then setup the worker threads to control the GPUs. After initialzeGPUs completes, gpu[] is filled out and
    ready to be used by the main program
*/
void ExecutionConfiguration::initializeGPUs(const std::vector<int>& gpu_ids, bool min_cpu)
	{
	// first check for some simple errors
	if (gpu_ids.size() == 0)
		{
		cout << endl << "***Error! No GPUs were specified!" << endl << endl;
		throw runtime_error("Error initializing execution configuration");
		}
		
	int capable_count = getNumCapableGPUs();
	if (capable_count == 0)
		{
		cout << endl << "***Error! No capable GPUs were found!" << endl << endl;
		throw runtime_error("Error initializing execution configuration");
		}
		
	// setup the flags
	int flags = 0;
	if (min_cpu)
		{
		if (CUDART_VERSION < 2020)
			cout << endl << "***Warning! --minimize-cpu-usage will have no effect because this hoomd was built "
			     << "against a version of CUDA prior to 2.2" << endl << endl;

		flags = cudaDeviceBlockingSync;
		}
	
	// determine how many automatic GPUs were requested
	int automatic_gpu_count = 0;
	
	// if we get here, at least one GPU is in the list to initialize and there is at least one capable GPU in the system
	// move on and individually check that each GPU is valid and available
	for (unsigned int i = 0; i < gpu_ids.size(); i++)
		{
		int gpu_id = gpu_ids[i];
		
		if (gpu_id < -1)
			{
			cout << endl << "***Error! The specified GPU id (" << gpu_id << ") is invalid." << endl << endl;
			throw runtime_error("Error initializing execution configuration");
			}
			
		if (gpu_id == -1)
			automatic_gpu_count++;
		else
		if ((unsigned int)gpu_id >= getNumTotalGPUs())
			{
			cout << endl << "***Error! The specified GPU id (" << gpu_id << ") is not present in the system." << endl
			     << "CUDA reports only " << getNumTotalGPUs() << endl << endl;
			throw runtime_error("Error initializing execution configuration");
			}
		
		if (!isGPUAvailable(gpu_ids[i]))
			{
			cout << endl << "***Error! The specified GPU id (" << gpu_id << ") is not available for executing HOOMD." 
			     << endl << "See the notice printed above to determine the reason." << endl << endl;
			throw runtime_error("Error initializing execution configuration");
			}
			
		// if we get here, everything checked out and the GPU can be initialized and added
		gpu.push_back(shared_ptr<GPUWorker>(new GPUWorker(gpu_id, flags, &m_gpu_list[0], m_gpu_list.size())));
		}
		
	if (automatic_gpu_count > 1 && !m_system_compute_exclusive)
		{
		cout << "Notice: More than one GPU was automatically chosen, but this sytem is not configured with all GPUs" 
		<< endl << "        set to compute-exclusive mode. This will likely result in strange behavior."
		<< endl;
		}
		
	printGPUStats();
	}

/*! Simply loops through all of the chosen GPUs and prints out a line of stats on them
	\pre gpu[] must be initialized and all worker threads created
*/
void ExecutionConfiguration::printGPUStats()
	{
	cout << "HOOMD is running on the following GPUs:" << endl;
	
	tagAll(__FILE__, __LINE__);
	for (unsigned int i = 0; i < gpu.size(); i++)
		{
		// get the properties for this device
		int dev;
		gpu[i]->call(bind(cudaGetDevice, &dev));
		cudaDeviceProp dev_prop;
		gpu[i]->call(bind(cudaGetDeviceProperties, &dev_prop, dev));
		
		// build a status line
		ostringstream s;
		
		// start with the device ID and name
		s << " [" << dev << "]";
		s << setw(22) << dev_prop.name;
		
		// then print the SM count and version
		s << setw(4) << dev_prop.multiProcessorCount << " SM_" << dev_prop.major << "." << dev_prop.minor;
		
		// and the clock rate
		float ghz = float(dev_prop.clockRate)/1e6;
		s.precision(3);
		s.fill('0');
		s << " @ " << setw(4) << ghz << " GHz";
		s.fill(' ');
		
		// and the total amount of memory
		int mib = int(float(dev_prop.totalGlobalMem) / float(1024*1024));
		s << ", " << setw(4) << mib << " MiB DRAM";
		
		// follow up with some flags to signify device features
		#if CUDART_VERSION > 2010
		if (dev_prop.kernelExecTimeoutEnabled)
			s << ", DIS";
		#endif
		
		cout << s.str() << endl;
		}
	}

//! Element in a priority sort of GPUs
struct gpu_elem
	{
	//! Constructor
	gpu_elem(float p=0.0f, int g=0) : priority(p), gpu_id(g) {}
	float priority;    //!< determined priority of the GPU
	int gpu_id;        //!< ID of the GPU
	};
	
//! less than operator for sorting gpu_elem
/*! \param a first element in the comparison
    \param b second element in the comparison
*/
bool operator<(const gpu_elem& a, const gpu_elem& b)
	{
	if (a.priority == b.priority)
		return a.gpu_id < b.gpu_id;
	else
		return a.priority > b.priority;
	}

/*! \param ignore_display If set to true, try to ignore GPUs attached to the display
    Each GPU that CUDA reports to exist is scrutinized to determine if it is actually capable of running HOOMD
    When one is found to be lacking, it is marked as unavailable and a short notice is printed as to why.

    \post m_gpu_list, m_gpu_available and m_system_compute_exclusive are all filled out
*/
void ExecutionConfiguration::scanGPUs(bool ignore_display)
	{
	#if CUDART_VERSION >= 2020
	// check the CUDA driver version
	int driverVersion = 0;
	cudaDriverGetVersion(&driverVersion);

	#ifndef _DEVICEEMU
	// device emulation mode doesn't need a driver

	// first handle the situation where no driver is installed (or it is a CUDA 2.1 or earlier driver)
	if (driverVersion == 0)
		{
		cout << endl << "***Warning! NVIDIA driver not installed or is too old, ignoring any GPUs in the system." 
		     << endl << endl;
		return;
		}
		
	// next, check to see if the driver is capable of running the version of CUDART that HOOMD was compiled against
	if (driverVersion < CUDART_VERSION)
		{
		int driver_major = driverVersion / 1000;
		int driver_minor = (driverVersion - driver_major * 1000) / 10;
		int cudart_major = CUDART_VERSION / 1000;
		int cudart_minor = (CUDART_VERSION - cudart_major * 1000) / 10;
		
		cout << endl << "***Warning! The NVIDIA driver only supports CUDA versions up to " << driver_major << "."
		<< driver_minor << ", but HOOMD was built against CUDA " << cudart_major << "." << cudart_minor << endl;
		cout << "            Ignoring any GPUs in the system." << endl;
		return;
		}
	#endif
	#endif
	
	// determine the number of GPUs that CUDA thinks there is
	int dev_count;
	cudaError_t error = cudaGetDeviceCount(&dev_count);
	if (error != cudaSuccess)
		{
		cerr << endl << "***Error! Error calling cudaGetDeviceCount()." << endl << endl;
		throw runtime_error("Error initializing execution configuration");
		}
	
	// initialize variables
	int n_exclusive_gpus = 0;
	m_gpu_available.resize(dev_count);
	
	// loop through each GPU and check it's properties
	for (int dev = 0; dev < dev_count; dev++)
		{
		// get the device properties
		cudaDeviceProp dev_prop;
		cudaError_t error = cudaGetDeviceProperties(&dev_prop, dev);
		if (error != cudaSuccess)
			{
			cerr << endl << "***Error! Error calling cudaGetDeviceProperties()." << endl << endl;
			throw runtime_error("Error initializing execution configuration");
			}
			
		// start by assuming that the device is available, it will be excluded later if it is not
		m_gpu_available[dev] = true;
			
		// if this is not a device emulation build: exclude the device emulation device
		#ifndef _DEVICEEMU
		if (dev_prop.major == 9999 && dev_prop.minor == 9999)
			{
			m_gpu_available[dev] = false;
			cout << "Notice: GPU id " << dev << " is not available for computation because "
			     << "it is an emulated device" << endl;
			}
		#endif
		
		// exclude a GPU if it's compute version is not high enough
		int compoundComputeVer = dev_prop.minor + dev_prop.major * 10;
		if (m_gpu_available[dev] && compoundComputeVer < CUDA_ARCH)
			{
			m_gpu_available[dev] = false;
			cout << "Notice: GPU id " << dev << " is not available for computation because "
			     << "it's compute capability is not high enough" << endl;
			
			int min_major = CUDA_ARCH/10;
			int min_minor = CUDA_ARCH - min_major*10;
			
			cout << "        This build of hoomd was compiled for a minimum capability of of " << min_major << "." 
			<< min_minor << " but the GPU is only " << dev_prop.major << "." << dev_prop.minor << endl;
			}
		
		#if CUDART_VERSION > 2010
		// ignore the display gpu if that was requested
		if (m_gpu_available[dev] && ignore_display && dev_prop.kernelExecTimeoutEnabled)
			{
			m_gpu_available[dev] = false;
			cout << "Notice: GPU id " << dev << " is not available for computation because "
			     << "it appears to be attached to a display" << endl;
			}
		#else
		if (ignore_display)
			{
			cout << endl << "***Warning! --ignore-dispaly-gpu is innefective because this build of HOOMD was compiled"
			<< " against a CUDA version older than 2.1" << endl << endl;
			}
		#endif
		
		#if CUDART_VERSION >= 2020
		// exclude a gpu if it is compute-prohibited
		if (m_gpu_available[dev] && dev_prop.computeMode == cudaComputeModeProhibited)
			{
			m_gpu_available[dev] = false;
			cout << "Notice: GPU id " << dev << " is not available for computation because "
			     << "it is set in the compute-prohibited mode" << endl;
			}
			
		// count the number of compute-exclusive gpus
		if (m_gpu_available[dev] && dev_prop.computeMode == cudaComputeModeExclusive)
			n_exclusive_gpus++;
		#endif
		}
		
	std::vector<gpu_elem> gpu_priorities;
	for (int dev = 0; dev < dev_count; dev++)
		{
		if (m_gpu_available[dev])
			{
			cudaDeviceProp dev_prop;
			cudaError_t error = cudaGetDeviceProperties(&dev_prop, dev);
			if (error != cudaSuccess)
				{
				cout << endl << "***Error! Error calling cudaGetDeviceProperties()." << endl << endl;
				throw runtime_error("Error initializing execution configuration");
				}
				
			// calculate a simple priority: multiprocessors * clock = speed, then subtract a bit if the device is
			// attached to a display
			float priority = float(dev_prop.clockRate * dev_prop.multiProcessorCount) / float(1e7);
			#if CUDART_VERSION > 2010
			if (dev_prop.kernelExecTimeoutEnabled)
				priority -= 0.1f;
			#endif
				
			gpu_priorities.push_back(gpu_elem(priority, dev));
			}
		}
			
	// sort the GPUs based on priority
	sort(gpu_priorities.begin(), gpu_priorities.end());
	// add the prioritized GPUs to the list
	for (unsigned int i = 0; i < gpu_priorities.size(); i++)
		{
		m_gpu_list.push_back(gpu_priorities[i].gpu_id);
		}
	
	// the system is fully compute-exclusive if all capable GPUs are compute-exclusive
	if (n_exclusive_gpus == getNumCapableGPUs())
		m_system_compute_exclusive = true;
	else
		m_system_compute_exclusive = false;
	}
		
	
/*! \param gpu_id ID of the GPU to check for availability
    \pre scanGPUs() has been called

    \return The availability statis of GPU \a gpu_id as determined by scanGPU()
*/
bool ExecutionConfiguration::isGPUAvailable(int gpu_id)
	{
	if (gpu_id < -1)
		return false;
	if (gpu_id == -1)
		return true;
	if ((unsigned int)gpu_id >= m_gpu_available.size())
		return false;
		
	return m_gpu_available[gpu_id];
	}
		

/*! \pre scanGPUs() has been called
    \return The count of avaialble GPUs deteremined by scanGPUs
*/
int ExecutionConfiguration::getNumCapableGPUs()
	{
	int count = 0;
	for (unsigned int i = 0; i < m_gpu_available.size(); i++)
		{
		if (m_gpu_available[i])
			count++;
		}
	return count;
	}

#endif

void export_ExecutionConfiguration()
	{
	scope in_exec_conf = class_<ExecutionConfiguration, boost::shared_ptr<ExecutionConfiguration>, boost::noncopyable >
		("ExecutionConfiguration", init< bool, bool >())
		.def(init<ExecutionConfiguration::executionMode, bool, bool>())
		.def(init<vector<int>, bool, bool >())
		.def_readonly("exec_mode", &ExecutionConfiguration::exec_mode)
#ifdef ENABLE_CUDA
	    .def("getComputeCapability", &ExecutionConfiguration::getComputeCapability)
#endif
		;
		
	enum_<ExecutionConfiguration::executionMode>("executionMode")
		.value("GPU", ExecutionConfiguration::GPU)
		.value("CPU", ExecutionConfiguration::CPU)
	;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
