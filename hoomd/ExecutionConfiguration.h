// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#ifndef __EXECUTION_CONFIGURATION__
#define __EXECUTION_CONFIGURATION__

// ensure that HOOMDMath.h is the first thing included
#include "HOOMDMath.h"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include <vector>
#include <string>
#include <memory>

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#endif

#ifdef ENABLE_TBB
#include <tbb/tbb.h>
#endif

#include "Messenger.h"
#include "MemoryTraceback.h"

/*! \file ExecutionConfiguration.h
    \brief Declares ExecutionConfiguration and related classes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifdef ENABLE_CUDA
//! Forward declaration
class CachedAllocator;
#endif

// values used in measuring hoomd launch timing
extern unsigned int hoomd_launch_time, hoomd_start_time, hoomd_mpi_init_time;
extern bool hoomd_launch_timing;

//! Defines the execution configuration for the simulation
/*! \ingroup data_structs
    ExecutionConfiguration is a data structure needed to support the hybrid CPU/GPU code. It initializes the CUDA GPU
    (if requested), stores information about the GPU on which this simulation is executing, and the number of CPUs
    utilized in the CPU mode.

    The execution configuration is determined at the beginning of the run and must
    remain static for the entire run. It can be accessed from the ParticleData of the
    system. DO NOT construct additional execution configurations. Only one is to be created for each run.

    The execution mode is specified in exec_mode. This is only to be taken as a hint,
    different compute classes are free to fall back on CPU implementations if no GPU is available. However,
    <b>ABSOLUTELY NO</b> CUDA calls should be made if exec_mode is set to CPU - making a CUDA call will initialize a
    GPU context and will error out on machines that do not have GPUs. isCUDAEnabled() is a convenience function to
    interpret the exec_mode and test if CUDA calls can be made or not.
*/
struct PYBIND11_EXPORT ExecutionConfiguration
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
                           std::vector<int> gpu_id = std::vector<int>(),
                           bool min_cpu=false,
                           bool ignore_display=false,
                           std::shared_ptr<Messenger> _msg=std::shared_ptr<Messenger>(),
                           unsigned int n_ranks = 0
                           #ifdef ENABLE_MPI
                           , MPI_Comm hoomd_world=MPI_COMM_WORLD
                           #endif
                           );

    ~ExecutionConfiguration();

#ifdef ENABLE_MPI
    //! Returns the MPI communicator
    MPI_Comm getMPICommunicator() const
        {
        return m_mpi_comm;
        }
    //! Returns the HOOMD World MPI communicator
    MPI_Comm getHOOMDWorldMPICommunicator() const
        {
        return m_hoomd_world;
        }
#endif

    //! Guess local rank of this processor, used for GPU initialization
    /*! \returns Local rank guessed from common environment variables
     *           or falls back to the global rank if no information is available
     */
    int guessLocalRank();

    executionMode exec_mode;    //!< Execution mode specified in the constructor
    unsigned int n_cpu;         //!< Number of CPUS hoomd is executing on
    bool m_cuda_error_checking;                //!< Set to true if GPU error checking is enabled
    std::shared_ptr<Messenger> msg;          //!< Messenger for use in printing messages to the screen / log file

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

    //! Get the number of active GPUs
    unsigned int getNumActiveGPUs() const
        {
        #ifdef ENABLE_CUDA
        return m_gpu_id.size();
        #else
        return 0;
        #endif
        }

    #ifdef ENABLE_CUDA
    //! Get the IDs of the active GPUs
    const std::vector<unsigned int>& getGPUIds() const
        {
        return m_gpu_id;
        }

    void cudaProfileStart() const
        {
        for (int idev = m_gpu_id.size()-1; idev >= 0; idev--)
            {
            cudaSetDevice(m_gpu_id[idev]);
            cudaDeviceSynchronize();
            cudaProfilerStart();
            }
        }

    void cudaProfileStop() const
        {
        for (int idev = m_gpu_id.size()-1; idev >= 0; idev--)
            {
            cudaSetDevice(m_gpu_id[idev]);
            cudaDeviceSynchronize();
            cudaProfilerStop();
            }
        }
    #endif

    //! Sync up all active GPUs
    void multiGPUBarrier() const;

    //! Begin a multi-GPU section
    void beginMultiGPU() const;

    //! End a multi-GPU section
    void endMultiGPU() const;

    //! Get the name of the executing GPU (or the empty string)
    std::string getGPUName(unsigned int idev=0) const;

#ifdef ENABLE_CUDA
    //! Get the device properties of a logical GPU
    cudaDeviceProp getDeviceProperties(unsigned int idev) const
        {
        return m_dev_prop[idev];
        }
#endif

    bool allConcurrentManagedAccess() const
        {
        // return cached value
        return m_concurrent;
        }

#ifdef ENABLE_CUDA
    cudaDeviceProp dev_prop;              //!< Cached device properties of the first GPU
    std::vector<unsigned int> m_gpu_id;   //!< IDs of active GPUs
    std::vector<cudaDeviceProp> m_dev_prop; //!< Device configuration of active GPUs

    //! Get the compute capability of the GPU that we are running on
    std::string getComputeCapabilityAsString(unsigned int igpu = 0) const;

    //! Get the compute capability of the GPU
    unsigned int getComputeCapability(unsigned int igpu = 0) const;

    //! Handle cuda error message
    void handleCUDAError(cudaError_t err, const char *file, unsigned int line) const;
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

    //! Set the HOOMD world MPI communicator
    void setHOOMDWorldMPICommunicator(const MPI_Comm mpi_comm)
        {
        m_hoomd_world = mpi_comm;
        }

    //! Perform a job-wide MPI barrier
    void barrier()
        {
        MPI_Barrier(m_mpi_comm);
        }
    #else
    bool isRoot() const
        {
        return true;
        }
    #endif

    #ifdef ENABLE_TBB
    //! set number of TBB threads
    void setNumThreads(unsigned int num_threads)
        {
        m_task_scheduler.reset(new tbb::task_scheduler_init(num_threads));
        m_num_threads = num_threads;
        }
    #endif

    //! Return the number of active threads
    unsigned int getNumThreads() const
        {
        #ifdef ENABLE_TBB
        return m_num_threads;
        #else
        return 0;
        #endif
        }


    #ifdef ENABLE_CUDA
    //! Returns the cached allocator for temporary allocations
    const CachedAllocator& getCachedAllocator() const
        {
        return *m_cached_alloc;
        }

    //! Returns the cached allocator for temporary allocations
    const CachedAllocator& getCachedAllocatorManaged() const
        {
        return *m_cached_alloc_managed;
        }
    #endif

    //! Set up memory tracing
    void setMemoryTracing(bool enable)
        {
        if (enable)
            m_memory_traceback = std::unique_ptr<MemoryTraceback>(new MemoryTraceback);
        else
            m_memory_traceback = std::unique_ptr<MemoryTraceback>();
        }

    //! Returns the memory tracer
    const MemoryTraceback *getMemoryTracer() const
        {
        return m_memory_traceback.get();
        }

    //! Returns true if we are in a multi-GPU block
    bool inMultiGPUBlock() const
        {
        return m_in_multigpu_block;
        }

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

    std::vector< bool > m_gpu_available;    //!< true if the GPU is available for computation, false if it is not
    bool m_system_compute_exclusive;        //!< true if every GPU in the system is marked compute-exclusive
    std::vector< int > m_gpu_list;          //!< A list of capable GPUs listed in priority order
    std::vector< cudaEvent_t > m_events;      //!< A list of events to synchronize between GPUs
#endif
    bool m_concurrent;                      //!< True if all GPUs have concurrentManagedAccess flag

    mutable bool m_in_multigpu_block;       //!< Tracks whether we are in a multi-GPU block

#ifdef ENABLE_MPI
    void splitPartitions(const MPI_Comm mpi_comm); //!< Create partitioned communicators

    MPI_Comm m_mpi_comm;                   //!< The MPI communicator
    MPI_Comm m_hoomd_world;                //!< The HOOMD world communicator
    unsigned int m_n_rank;                 //!< Ranks per partition
#endif

    unsigned int m_rank;                   //!< Rank of this processor (0 if running in single-processor mode)

    #ifdef ENABLE_CUDA
    CachedAllocator *m_cached_alloc;       //!< Cached allocator for temporary allocations
    CachedAllocator *m_cached_alloc_managed; //!< Cached allocator for temporary allocations in managed memory
    #endif

    #ifdef ENABLE_TBB
    std::unique_ptr<tbb::task_scheduler_init> m_task_scheduler; //!< The TBB task scheduler
    unsigned int m_num_threads;            //!<  The number of TBB threads used
    #endif

    //! Setup and print out stats on the chosen CPUs/GPUs
    void setupStats();

    std::unique_ptr<MemoryTraceback> m_memory_traceback;    //!< Keeps track of allocations
    };

// Macro for easy checking of CUDA errors - enabled all the time
#ifdef ENABLE_CUDA
#define CHECK_CUDA_ERROR() { \
    cudaError_t err_sync = cudaGetLastError(); \
    this->m_exec_conf->handleCUDAError(err_sync, __FILE__, __LINE__); \
    auto gpu_map = this->m_exec_conf->getGPUIds(); \
    for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev) \
        { \
        cudaSetDevice(gpu_map[idev]); \
        cudaError_t err_async = cudaDeviceSynchronize(); \
        this->m_exec_conf->handleCUDAError(err_async, __FILE__, __LINE__); \
        } \
    }
#else
#define CHECK_CUDA_ERROR()
#endif

//! Exports ExecutionConfiguration to python
#ifndef NVCC
void export_ExecutionConfiguration(pybind11::module& m);
#endif

#endif
