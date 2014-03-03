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
// Maintainer: joaander

#ifndef __EXECUTION_CONFIGURATION__
#define __EXECUTION_CONFIGURATION__

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include "Messenger.h"

/*! \file ExecutionConfiguration.h
    \brief Declares ExecutionConfiguration and related classes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_CUDA
//! Forward declaration
class CachedAllocator;
#endif

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
struct ExecutionConfiguration : boost::noncopyable
    {
    //! Simple enum for the execution modes
    enum executionMode
        {
        GPU,    //!< Execute on the GPU
        CPU,    //!< Execute on the CPU
        AUTO,   //!< Auto select between GPU and CPU
        };

    //! Constructor
    ExecutionConfiguration(executionMode mode=AUTO,
                           int gpu_id=-1,
                           bool min_cpu=false,
                           bool ignore_display=false,
                           boost::shared_ptr<Messenger> _msg=boost::shared_ptr<Messenger>(),
                           unsigned int n_ranks = 0);

    ~ExecutionConfiguration();

#ifdef ENABLE_MPI
    //! Returns the boost MPI communicator
    const MPI_Comm getMPICommunicator() const
        {
        return m_mpi_comm;
        }
#endif

    //! Guess local rank of this processor, used for GPU initialization
    /*! \returns Local rank guessed from common environment variables
     *           or -1 if no information is available
     */
    static int guessLocalRank();

    executionMode exec_mode;    //!< Execution mode specified in the constructor
    unsigned int n_cpu;         //!< Number of CPUS hoomd is executing on
    bool m_cuda_error_checking;                //!< Set to true if GPU error checking is enabled
    boost::shared_ptr<Messenger> msg;          //!< Messenger for use in printing messages to the screen / log file

    //! Returns true if CUDA is enabled
    bool isCUDAEnabled() const
        {
        return (exec_mode == GPU);
        }

    //! Returns true if CUDA error checking is enabled
    bool isCUDAErrorCheckingEnabled() const
        {
        #ifndef NDEBUG
        return true;
        #else
        return m_cuda_error_checking;
        #endif
        }

    //! Sets the cuda error checking mode
    void setCUDAErrorChecking(bool cuda_error_checking)
        {
        m_cuda_error_checking = cuda_error_checking;
        }

    //! Get the name of the executing GPU (or the empty string)
    std::string getGPUName() const;
#ifdef ENABLE_CUDA
    cudaDeviceProp dev_prop;    //!< Cached device properties

    //! Get the compute capability of the GPU that we are running on
    std::string getComputeCapabilityAsString() const;

    //! Get thie compute capability of the GPU
    unsigned int getComputeCapability() const;

    //! Handle cuda error message
    void handleCUDAError(cudaError_t err, const char *file, unsigned int line) const;

    //! Check for cuda errors
    void checkCUDAError(const char *file, unsigned int line) const;
#endif

    //! Return the rank of this processor in the partition
    unsigned int getRank() const
        {
        return m_rank;
        }

    #ifdef ENABLE_MPI
    //! Return the global rank of this processor
    static unsigned int getRankGlobal()
        {
        int rank;
        // get rank on world communicator
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
        }

    //! Return the global communicator size
    static unsigned int getNRanksGlobal()
        {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
        }

    //! Returns the partition number of this processor
    unsigned int getPartition() const
        {
        return m_n_rank ? getRankGlobal()/m_n_rank : 0;
        }

    //! Returns the number of partitions
    unsigned int getNPartitions() const
        {
        return m_n_rank ? getNRanksGlobal()/m_n_rank : 1;
        }

    //! Return the number of ranks in this partition
    unsigned int getNRanks() const;

    //! Returns true if this is the root processor
    bool isRoot() const
        {
        return getRank() == 0;
        }

    //! Set the MPI communicator
    void setMPICommunicator(const MPI_Comm mpi_comm)
        {
        m_mpi_comm = mpi_comm;
        }
    #endif

    #ifdef ENABLE_CUDA
    //! Returns the cached allocator for temporary allocations
    const CachedAllocator& getCachedAllocator() const
        {
        return *m_cached_alloc;
        }
    #endif

private:
#ifdef ENABLE_CUDA
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
#endif

#ifdef ENABLE_MPI
    void initializeMPI();                  //!< Initialize MPI environment

    MPI_Comm m_mpi_comm;                   //!< The MPI communicator
    unsigned int m_n_rank;                 //!< Ranks per partition
#endif

    unsigned int m_rank;                   //!< Rank of this processor (0 if running in single-processor mode)

    #ifdef ENABLE_CUDA
    CachedAllocator *m_cached_alloc;       //!< Cached allocator for temporary allocations
    #endif

    //! Setup and print out stats on the chosen CPUs/GPUs
    void setupStats();
    };

// Macro for easy checking of CUDA errors - enabled all the time
#define CHECK_CUDA_ERROR() this->m_exec_conf->checkCUDAError(__FILE__, __LINE__);

//! Exports ExecutionConfiguration to python
void export_ExecutionConfiguration();

#endif
