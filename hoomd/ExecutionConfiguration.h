// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EXECUTION_CONFIGURATION__
#define __EXECUTION_CONFIGURATION__

// ensure that HOOMDMath.h is the first header included to work around broken mpi headers
#include "HOOMDMath.h"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include "MPIConfiguration.h"

#include <memory>
#include <string>
#include <vector>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#ifdef ENABLE_ROCTRACER
#ifdef __HIP_PLATFORM_HCC__
#include <roctracer/roctracer_ext.h>
#endif
#endif
#endif

#include "Messenger.h"

/*! \file ExecutionConfiguration.h
    \brief Declares ExecutionConfiguration and related classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hoomd
    {
#if defined(ENABLE_HIP)
//! Forward declaration
class CachedAllocator;
#endif

//! Defines the execution configuration for the simulation
/*! \ingroup data_structs
    ExecutionConfiguration is a data structure needed to support the hybrid CPU/GPU code. It
   initializes the CUDA GPU (if requested), stores information about the GPU on which this
   simulation is executing, and the number of CPUs utilized in the CPU mode.

    The execution configuration is determined at the beginning of the run and must
    remain static for the entire run. It can be accessed from the ParticleData of the
    system. DO NOT construct additional execution configurations. Only one is to be created for each
   run.

    The execution mode is specified in exec_mode. This is only to be taken as a hint,
    different compute classes are free to fall back on CPU implementations if no GPU is available.
   However, <b>ABSOLUTELY NO</b> CUDA calls should be made if exec_mode is set to CPU - making a
   CUDA call will initialize a GPU context and will error out on machines that do not have GPUs.
   isCUDAEnabled() is a convenience function to interpret the exec_mode and test if CUDA calls can
   be made or not.
*/
class PYBIND11_EXPORT ExecutionConfiguration
    {
    public:
    //! Simple enum for the execution modes
    enum executionMode
        {
        GPU,  //!< Execute on the GPU
        CPU,  //!< Execute on the CPU
        AUTO, //!< Auto select between GPU and CPU
        };

    //! Constructor
    ExecutionConfiguration(executionMode mode = AUTO,
                           int gpu_id = -1,
                           std::shared_ptr<MPIConfiguration> mpi_config
                           = std::shared_ptr<MPIConfiguration>(),
                           std::shared_ptr<Messenger> _msg = std::shared_ptr<Messenger>());

    ~ExecutionConfiguration();

    //! Returns the MPI Configuration
    std::shared_ptr<MPIConfiguration> getMPIConfig() const
        {
        assert(m_mpi_config);
        return m_mpi_config;
        }

#ifdef ENABLE_MPI
    //! Returns the MPI communicator
    MPI_Comm getMPICommunicator() const
        {
        assert(m_mpi_config);
        return m_mpi_config->getCommunicator();
        }

    //! Returns the HOOMD World MPI communicator
    MPI_Comm getHOOMDWorldMPICommunicator() const
        {
        assert(m_mpi_config);
        return m_mpi_config->getHOOMDWorldCommunicator();
        }
#endif

    std::shared_ptr<Messenger>
        msg; //!< Messenger for use in printing messages to the screen / log file

    //! Returns true if CUDA is enabled
    bool isCUDAEnabled() const
        {
        return (exec_mode == GPU);
        }

    //! Returns true if CUDA error checking is enabled
    bool isCUDAErrorCheckingEnabled() const
        {
        return m_hip_error_checking;
        }

    //! Sets the hip error checking mode
    void setCUDAErrorChecking(bool hip_error_checking)
        {
        m_hip_error_checking = hip_error_checking;
        }

    /// Select the active GPU
    const void setDevice() const
        {
#if defined(ENABLE_HIP)
        hipSetDevice(m_gpu_id);
#endif
        }

    /// get the device id
    const unsigned int getGPUId() const
        {
#if defined(ENABLE_HIP)
        return m_gpu_id;
#else
        return 0;
#endif
        }

#if defined(ENABLE_HIP)
    void hipProfileStart() const
        {
        hipSetDevice(m_gpu_id);
        hipDeviceSynchronize();

#ifdef __HIP_PLATFORM_NVCC__
        hipProfilerStart();
#elif defined(__HIP_PLATFORM_HCC__)
#ifdef ENABLE_ROCTRACER
        roctracer_start();
#else
        msg->warning() << "ROCtracer not enabled, profile start/stop not available" << std::endl;
#endif
#endif
        }

    void hipProfileStop() const
        {
        hipSetDevice(m_gpu_id);
        hipDeviceSynchronize();
#ifdef __HIP_PLATFORM_NVCC__
        hipProfilerStop();
#elif defined(__HIP_PLATFORM_HCC__)
#ifdef ENABLE_ROCTRACER
        roctracer_stop();
#else
        msg->warning() << "ROCtracer not enabled, profile start/stop not available" << std::endl;
#endif
#endif
        }
#endif

    bool allConcurrentManagedAccess() const
        {
        return false;
        }

#ifdef ENABLE_HIP
    hipDeviceProp_t dev_prop; //!< Cached device properties of the first GPU

    /// Compute capability of the GPU formatted as a tuple (major, minor)
    std::pair<unsigned int, unsigned int> getComputeCapability() const;

    //! Handle hip error message
    void handleHIPError(hipError_t err, const char* file, unsigned int line) const;
#endif

    /*
     * The following MPI related methods only wrap those of the MPIConfiguration object,
       which can obtained with getMPIConfig(), and are provided as a legacy API.
    */

    //! Return the rank of this processor in the partition
    unsigned int getRank() const
        {
        assert(m_mpi_config);
        return m_mpi_config->getRank();
        }

    //! Returns the partition number of this processor
    unsigned int getPartition() const
        {
        assert(m_mpi_config);
        return m_mpi_config->getPartition();
        }

    //! Returns the number of partitions
    unsigned int getNPartitions() const
        {
        assert(m_mpi_config);
        return m_mpi_config->getNPartitions();
        }

    //! Return the number of ranks in this partition
    unsigned int getNRanks() const
        {
        assert(m_mpi_config);
        return m_mpi_config->getNRanks();
        }

    //! Returns true if this is the root processor
    bool isRoot() const
        {
        assert(m_mpi_config);
        return m_mpi_config->isRoot();
        }

#if defined(ENABLE_HIP)
    //! Returns the cached allocator for temporary allocations
    CachedAllocator& getCachedAllocator() const
        {
        return *m_cached_alloc;
        }

    //! Returns the cached allocator for temporary allocations
    CachedAllocator& getCachedAllocatorManaged() const
        {
        return *m_cached_alloc_managed;
        }
#endif

    //! Set up memory tracing
    void setMemoryTracing(bool enable)
        {
        m_memory_tracing = enable;
        }

    bool memoryTracingEnabled() const
        {
        return m_memory_tracing;
        }

    /// Get a list of the capable devices
    static std::vector<std::string> getCapableDevices()
        {
#ifdef ENABLE_HIP
        scanGPUs();
#endif
        return s_capable_gpu_descriptions;
        }

    /// Get a list of the capable devices
    static std::vector<std::string> getScanMessages()
        {
#ifdef ENABLE_HIP
        scanGPUs();
#endif
        return s_gpu_scan_messages;
        }

    /// Get the active device
    std::string getActiveDevice()
        {
        return m_active_device_description;
        }

    private:
    //! Guess local rank of this processor, used for GPU initialization
    /*! \returns Local rank guessed from common environment variables
                 or falls back to the global rank if no information is available
     */
    int guessLocalRank();

#if defined(ENABLE_HIP)
    //! Initialize the GPU with the given id (where gpu_id is an index into s_capable_gpu_ids)
    void initializeGPU(int gpu_id);

    /// Provide a string that describes a GPU device
    static std::string describeGPU(int id, hipDeviceProp_t prop);

    /** Scans through all GPUs reported by CUDA and marks if they are available

        Determine which GPUs are available for use by HOOMD.

        @post Populate s_gpu_scan_complete, s_gpu_scan_messages, s_gpu_list, and
        s_capable_gpu_descriptions.
    */
    static void scanGPUs();

    /// ID of active GPU
    unsigned int m_gpu_id;

    /// Device configuration of active GPU
    hipDeviceProp_t m_dev_prop;
#endif

    /// Execution mode
    executionMode exec_mode;

    /// True when GPU error checking is enabled
    bool m_hip_error_checking;

    /// The MPI configuration
    std::shared_ptr<MPIConfiguration> m_mpi_config;

    /// Set to true
    static bool s_gpu_scan_complete;

    /// Status messages generated during the device scan
    static std::vector<std::string> s_gpu_scan_messages;

    /// List of the capable device IDs
    static std::vector<int> s_capable_gpu_ids;

    /// Description of the GPU devices
    static std::vector<std::string> s_capable_gpu_descriptions;

    /// Description of the active device
    std::string m_active_device_description;

#if defined(ENABLE_HIP)
    std::unique_ptr<CachedAllocator> m_cached_alloc; //!< Cached allocator for temporary allocations
    std::unique_ptr<CachedAllocator>
        m_cached_alloc_managed; //!< Cached allocator for temporary allocations in managed memory
#endif

    //! Setup and print out stats on the chosen CPUs/GPUs
    void setupStats();

    bool m_memory_tracing = false;
    };

#if defined(ENABLE_HIP)
#define CHECK_CUDA_ERROR()                                                \
        {                                                                 \
        hipError_t err_sync = hipPeekAtLastError();                       \
        this->m_exec_conf->handleHIPError(err_sync, __FILE__, __LINE__);  \
        this->m_exec_conf->setDevice();                                   \
        hipError_t err_async = hipDeviceSynchronize();                    \
        this->m_exec_conf->handleHIPError(err_async, __FILE__, __LINE__); \
        }
#else
#define CHECK_CUDA_ERROR()
#endif

namespace detail
    {
//! Exports ExecutionConfiguration to python
#ifndef __HIPCC__
void export_ExecutionConfiguration(pybind11::module& m);
#endif
    } // end namespace detail

    } // end namespace hoomd

#endif
