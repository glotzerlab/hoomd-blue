/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// $Id$
// $URL$
// Maintainer: joaander

#ifndef __EXECUTION_CONFIGURATION__
#define __EXECUTION_CONFIGURATION__

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>

/*! \file ExecutionConfiguration.h
    \brief Declares ExecutionConfiguration and related classes
*/

//! Defines the execution configuration for the simulation
/*! \ingroup data_structs
    ExecutionConfiguration is a data structure needed to support the hybrid CPU/GPU code. It initializes the CUDA GPU
	(if requested), stores information about the GPU on which this simulation is executing, and the number of CPUs
	utilized in the CPU mode. 

    The execution configuration is determined at the beginning of the run and must
    remain static for the entire run. It can be accessed from the ParticleData of the
    system. DO NOT construct additional exeuction configurations. Only one is to be created for each run.

    The execution mode is specified in exec_mode. This is only to be taken as a hint,
    different compute classes are free to fall back on CPU implementations if no GPU is available. However,
	<b>ABSOLUTELY NO</b> CUDA calls should be made if exec_mode is set to CPU - making a CUDA call will initialize a
	GPU context and will error out on machines that do not have GPUs. isCUDAEnabled() is a convenience function to
	interpret the exec_mode and test if CUDA calls can be made or not.
*/
struct ExecutionConfiguration
    {
    //! Simple enum for the execution modes
    enum executionMode
        {
        GPU,    //!< Execute on the GPU
        CPU //!< Execute on the CPU
        };
        
    //! Default constructor
    ExecutionConfiguration(bool min_cpu=false, bool ignore_display=false);
    
    //! Force a mode selection
    ExecutionConfiguration(executionMode mode, int gpu_id=-1, bool min_cpu=false, bool ignore_display=false);
    
	executionMode exec_mode;    //!< Execution mode specified in the constructor
    unsigned int n_cpu;         //!< Number of CPUS hoomd is executing on
    
#ifdef ENABLE_CUDA
	cudaDeviceProp dev_prop;	//!< Cached device properties
    
	//! Returns true if CUDA is enabled
	bool isCUDAEnabled() const
		{
		return (exec_mode == GPU);
		}
	
	//! Returns true if CUDA error checking is enabled
	bool isCUDAErrorCheckingEnabled() const
		{
		return m_cuda_error_checking;
		}
	
	//! Sets the cuda error checking mode
	void setCUDAErrorChecking(bool cuda_error_checking)
		{
		m_cuda_error_checking = cuda_error_checking;
		}
	
    //! Get the compute capability of the GPU that we are running on
    std::string getComputeCapability();
	
	//! Handle cuda error message
	static void handleCUDAError(cudaError_t err, const char *file, unsigned int line);
	
	//! Check for cuda errors
	static void checkCUDAError(const char *file, unsigned int line);
    
private:
    //! Initialize the GPU with the given id
    void initializeGPU(int gpu_id, bool min_cpu);
    
    //! Print out stats on the chosen GPUs
    void printGPUStats();
    
    //! Scans through all GPUs reported by CUDA and marks if they are available
    void scanGPUs(bool ignore_display);
    
    //! Returns true if the given GPU is available for computation
    bool isGPUAvailable(int gpu_id);
    
    //! Returns the count of capable GPUs
    int getNumCapableGPUs();
    
    //! Return the number of GPUs that can be checked for availability
    unsigned int getNumTotalGPUs()
        {
        return (unsigned int)m_gpu_available.size();
        }
        
    std::vector< bool > m_gpu_available;    //!< true if the GPU is avaialble for computation, false if it is not
    bool m_system_compute_exclusive;        //!< true if every GPU in the system is marked compute-exclusive
    std::vector< int > m_gpu_list;          //!< A list of capable GPUs listed in priority order
	bool m_cuda_error_checking;				//!< Set to true if GPU error checking is enabled
#endif
    
    //! Setup and print out stats on the chosen CPUs/GPUs
    void setupStats();
    };

//! Macro for easy checking of CUDA errors
#define CHECK_CUDA_ERROR() ExecutionConfiguration::checkCUDAError(__FILE__, __LINE__);

//! Exports ExecutionConfiguration to python
void export_ExecutionConfiguration();

#endif

