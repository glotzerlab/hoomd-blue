// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "ExecutionConfiguration.h"
#include "HOOMDVersion.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif
namespace py = pybind11;

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>

using namespace std;

#ifdef ENABLE_CUDA
#include "CachedAllocator.h"
#endif

#ifdef ENABLE_TBB
std::unique_ptr<tbb::global_control> ExecutionConfiguration::tbb_thread_control;
#endif

/*! \file ExecutionConfiguration.cc
    \brief Defines ExecutionConfiguration and related classes
*/

/*! \param mode Execution mode to set (cpu or gpu)
    \param gpu_id List of GPU IDs on which to run, or empty for automatic selection
    \param min_cpu If set to true, cudaDeviceBlockingSync is set to keep the CPU usage of HOOMD to a minimum
    \param ignore_display If set to true, try to ignore GPUs attached to the display
    \param mpi_config MPI configuration object
    \param _msg Messenger to use for status message printing

    Explicitly force the use of either CPU or GPU execution. If GPU execution is selected, then a default GPU choice
    is made by not calling cudaSetDevice.
*/
ExecutionConfiguration::ExecutionConfiguration(executionMode mode,
                                               std::vector<int> gpu_id,
                                               bool min_cpu,
                                               bool ignore_display,
                                               std::shared_ptr<MPIConfiguration> mpi_config,
                                               std::shared_ptr<Messenger> _msg
                                               )
    : m_cuda_error_checking(false), m_mpi_config(mpi_config), msg(_msg)
    {
    if (! m_mpi_config)
        {
        // create mpi config internally
        m_mpi_config = std::shared_ptr<MPIConfiguration>(new MPIConfiguration());
        }

    if (!msg)
        {
        // create Messenger internally
        msg = std::shared_ptr<Messenger>(new Messenger(m_mpi_config));
        }

    ostringstream s;
    for (auto it = gpu_id.begin(); it != gpu_id.end(); ++it)
        {
        s << *it << " ";
        }

    msg->notice(5) << "Constructing ExecutionConfiguration: ( " << s.str() << ") " <<  min_cpu << " " << ignore_display << endl;
    exec_mode = mode;

#ifdef ENABLE_CUDA
    // scan the available GPUs
    scanGPUs(ignore_display);
    int dev_count = getNumCapableGPUs();

    // auto select a mode
    if (exec_mode == AUTO)
        {
        // if there are available GPUs, initialize them. Otherwise, default to running on the CPU
        if (dev_count > 0)
            exec_mode = GPU;
        else
            exec_mode = CPU;
        }

    m_concurrent = exec_mode==GPU;
    m_in_multigpu_block = false;

    // now, exec_mode should be either CPU or GPU - proceed with initialization

    // initialize the GPU if that mode was requested
    if (exec_mode == GPU)
        {
        bool found_local_rank = false;
        int local_rank = guessLocalRank(found_local_rank);
        if (!gpu_id.size() && found_local_rank)
            {
            // if we found a local rank, use that to select the GPU
            gpu_id.push_back((local_rank % dev_count));
            }

        cudaSetValidDevices(&m_gpu_list[0], (int)m_gpu_list.size());

        if (! gpu_id.size())
            {
            // auto-detect a single GPU
            initializeGPU(-1, min_cpu);
            }
        else
            {
            // initialize all requested GPUs
            for (auto it = gpu_id.begin(); it != gpu_id.end(); ++it)
                initializeGPU(*it, min_cpu);
            }
        }
#else
    if (exec_mode == GPU)
        {
        msg->error() << "GPU execution requested, but this hoomd was built without CUDA support" << endl;
        throw runtime_error("Error initializing execution configuration");
        }
    // "auto-select" the CPU
    exec_mode = CPU;
    m_concurrent = false;
#endif

    setupStats();

    #ifdef ENABLE_CUDA
    if (exec_mode == GPU)
        {
        if (! m_concurrent && gpu_id.size() > 1)
            {
            msg->errorAllRanks() << "Multi-GPU execution requested, but not all GPUs support concurrent managed access" << endl;
            throw runtime_error("Error initializing execution configuration");
            }

        #ifndef ALWAYS_USE_MANAGED_MEMORY
        // disable managed memory when running on single GPU
        if (m_gpu_id.size() == 1)
            {
            m_concurrent = false;
            }
        #endif

        if (m_concurrent)
            {
            // compare compute capabilities
            for (unsigned int idev = 0; idev < gpu_id.size(); ++idev)
                {
                if (m_dev_prop[idev].major != m_dev_prop[0].major
                    || m_dev_prop[idev].minor != m_dev_prop[0].minor)
                    {
                    // the autotuner may pick up different block sizes for different GPUs
                    msg->warning() << "Multi-GPU execution requested, but GPUs have differing compute capabilities" << endl;
                    msg->warning() << "Continuing anyways, but autotuner may not work correctly and simulation may crash." << endl;
                    }
                }
            }

        // select first device by default
        cudaSetDevice(m_gpu_id[0]);

        cudaError_t err_sync = cudaGetLastError();
        handleCUDAError(err_sync, __FILE__, __LINE__);

        // initialize cached allocator, max allocation 0.5*global mem
        m_cached_alloc.reset(new CachedAllocator(false, (unsigned int)(0.5f*(float)dev_prop.totalGlobalMem)));
        m_cached_alloc_managed.reset(new CachedAllocator(true, (unsigned int)(0.5f*(float)dev_prop.totalGlobalMem)));
        }
    #endif

    #ifdef ENABLE_MPI
    // ensure that all ranks are on the same execution configuration
    if (getNRanks() > 1)
        {
        executionMode rank0_mode = exec_mode;
        bcast(rank0_mode, 0, m_mpi_config->getCommunicator());

        // ensure that all ranks terminate here
        int errors = 0;
        if (rank0_mode != exec_mode)
            errors = 1;

        MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, m_mpi_config->getCommunicator());

        if (errors != 0)
            {
            msg->error() << "Not all ranks have the same execution context (some are CPU and some are GPU)" << endl;
            throw runtime_error("Error initializing execution configuration");
            }
        }

    if (hoomd_launch_timing && m_mpi_config->getNRanksGlobal() > 1)
        {
        // compute the number of seconds to get an exec conf
        timeval t;
        gettimeofday(&t, NULL);
        unsigned int conf_time = t.tv_sec - hoomd_launch_time;

        // get the min and max times
        unsigned int start_time_min, start_time_max, mpi_init_time_min, mpi_init_time_max, conf_time_min, conf_time_max;
        MPI_Reduce(&hoomd_start_time, &start_time_min, 1, MPI_UNSIGNED, MPI_MIN, 0, m_mpi_config->getHOOMDWorldCommunicator());
        MPI_Reduce(&hoomd_start_time, &start_time_max, 1, MPI_UNSIGNED, MPI_MAX, 0, m_mpi_config->getHOOMDWorldCommunicator());

        MPI_Reduce(&hoomd_mpi_init_time, &mpi_init_time_min, 1, MPI_UNSIGNED, MPI_MIN, 0, m_mpi_config->getHOOMDWorldCommunicator());
        MPI_Reduce(&hoomd_mpi_init_time, &mpi_init_time_max, 1, MPI_UNSIGNED, MPI_MAX, 0, m_mpi_config->getHOOMDWorldCommunicator());

        MPI_Reduce(&conf_time, &conf_time_min, 1, MPI_UNSIGNED, MPI_MIN, 0, m_mpi_config->getHOOMDWorldCommunicator());
        MPI_Reduce(&conf_time, &conf_time_max, 1, MPI_UNSIGNED, MPI_MAX, 0, m_mpi_config->getHOOMDWorldCommunicator());

        // write them out to a file
        if (m_mpi_config->getRankGlobal() == 0)
            {
            msg->notice(2) << "start_time:    [" << start_time_min << ", " << start_time_max << "]" << std::endl;
            msg->notice(2) << "mpi_init_time: [" << mpi_init_time_min << ", " << mpi_init_time_max << "]" << std::endl;
            msg->notice(2) << "conf_time:     [" << conf_time_min << ", " << conf_time_max << "]" << std::endl;
            }
        }
    #endif

    #ifdef ENABLE_TBB
    m_num_threads = std::thread::hardware_concurrency();

    char *env;
    if ((env = getenv("OMP_NUM_THREADS")) != NULL)
        {
        unsigned int num_threads = atoi(env);
        msg->notice(2) << "Setting number of TBB threads to value of OMP_NUM_THREADS=" << num_threads << std::endl;
        setNumThreads(num_threads);
        }
    #endif

    #ifdef ENABLE_CUDA
    // setup synchronization events
    m_events.resize(m_gpu_id.size());
    for (int idev = m_gpu_id.size()-1; idev >= 0; --idev)
        {
        cudaSetDevice(m_gpu_id[idev]);
        cudaEventCreateWithFlags(&m_events[idev],cudaEventDisableTiming);
        }
    #endif
    }

ExecutionConfiguration::~ExecutionConfiguration()
    {
    msg->notice(5) << "Destroying ExecutionConfiguration" << endl;

    #ifdef ENABLE_CUDA
    for (int idev = m_gpu_id.size()-1; idev >= 0; --idev)
        {
        cudaEventDestroy(m_events[idev]);
        }
    #endif

    #ifdef ENABLE_CUDA
    // the destructors of these objects can issue cuda calls, so free them before the device reset
    m_cached_alloc.reset();
    m_cached_alloc_managed.reset();
    #endif
    }

std::string ExecutionConfiguration::getGPUName(unsigned int idev) const
    {
    #ifdef ENABLE_CUDA
    if (exec_mode == GPU)
        return string(m_dev_prop[idev].name);
    else
        return string();
    #else
    return string();
    #endif
    }


#ifdef ENABLE_CUDA
/*! \returns Compute capability of GPU 0 as a string
    \note Silently returns an empty string if no GPUs are specified
*/
std::string ExecutionConfiguration::getComputeCapabilityAsString(unsigned int idev) const
    {
    ostringstream s;

    if (exec_mode == GPU)
        {
        s << m_dev_prop[idev].major << "." << m_dev_prop[idev].minor;
        }

    return s.str();
    }

/*! \returns Compute capability of the GPU formatted as 210 (for compute 2.1 as an example)
    \note Silently returns 0 if no GPU is being used
*/
unsigned int ExecutionConfiguration::getComputeCapability(unsigned int idev) const
    {
    unsigned int result = 0;

    if (exec_mode == GPU)
        {
        result = m_dev_prop[idev].major * 100 + m_dev_prop[idev].minor * 10;
        }

    return result;
    }

void ExecutionConfiguration::handleCUDAError(cudaError_t err, const char *file, unsigned int line) const
    {
    // if there was an error
    if (err != cudaSuccess)
        {
        // remove HOOMD_SOURCE_DIR from the front of the file
        if (strlen(file) > strlen(HOOMD_SOURCE_DIR))
            file += strlen(HOOMD_SOURCE_DIR);

        // print an error message
        msg->errorAllRanks() << string(cudaGetErrorString(err)) << " before "
                             << file << ":" << line << endl;

        // throw an error exception
        throw(runtime_error("CUDA Error"));
        }
    }

/*! \param gpu_id Index for the GPU to initialize, set to -1 for automatic selection
    \param min_cpu If set to true, the cudaDeviceBlockingSync device flag is set

    \pre scanGPUs has been called

    initializeGPU will loop through the specified list of GPUs, validate that each one is available for CUDA use
    and then setup CUDA to use the given GPU. After initializeGPU completes, cuda calls can be made by the main
    application.
*/
void ExecutionConfiguration::initializeGPU(int gpu_id, bool min_cpu)
    {
    int capable_count = getNumCapableGPUs();
    if (capable_count == 0)
        {
        msg->errorAllRanks() << "No capable GPUs were found!" << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    // setup the flags
    int flags = 0;
    if (min_cpu)
        {
        flags |= cudaDeviceBlockingSync;
        }
    else
        {
        flags |= cudaDeviceScheduleSpin;
        }

    if (gpu_id < -1)
        {
        msg->errorAllRanks() << "The specified GPU id (" << gpu_id << ") is invalid." << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    if (gpu_id >= (int)getNumTotalGPUs())
        {
        msg->errorAllRanks() << "The specified GPU id (" << gpu_id << ") is not present in the system." << endl;
        msg->errorAllRanks() << "CUDA reports only " << getNumTotalGPUs() << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    if (!isGPUAvailable(gpu_id))
        {
        msg->errorAllRanks() << "The specified GPU id (" << gpu_id << ") is not available for executing HOOMD." << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    cudaSetDeviceFlags(flags | cudaDeviceMapHost);

    if (gpu_id != -1)
        {
        cudaSetDevice(m_gpu_list[gpu_id]);
        }
    else
        {
        // initialize the default CUDA context
        cudaFree(0);
        }

    int cuda_gpu_id;
    cudaGetDevice(&cuda_gpu_id);

    // add to list of active GPUs
    m_gpu_id.push_back(cuda_gpu_id);

    cudaError_t err_sync = cudaGetLastError();
    handleCUDAError(err_sync, __FILE__, __LINE__);
    }

/*! Prints out a status line for the selected GPU
*/
void ExecutionConfiguration::printGPUStats()
    {
    msg->notice(1) << "HOOMD-blue is running on the following GPU(s):" << endl;

    // build a status line
    ostringstream s;

    for (unsigned int idev = 0; idev < m_gpu_id.size(); ++idev)
        {
        // start with the device ID and name
        unsigned int dev = m_gpu_id[idev];

        s << " [" << dev << "]";
        s << setw(22) << m_dev_prop[idev].name;

        // then print the SM count and version
        s << setw(4) << m_dev_prop[idev].multiProcessorCount << " SM_" << m_dev_prop[idev].major << "." << m_dev_prop[idev].minor;

        // and the clock rate
        float ghz = float(m_dev_prop[idev].clockRate)/1e6;
        s.precision(3);
        s.fill('0');
        s << " @ " << setw(4) << ghz << " GHz";
        s.fill(' ');

        // and the total amount of memory
        int mib = int(float(m_dev_prop[idev].totalGlobalMem) / float(1024*1024));
        s << ", " << setw(4) << mib << " MiB DRAM";

        // follow up with some flags to signify device features
        if (m_dev_prop[idev].kernelExecTimeoutEnabled)
            s << ", DIS";

        // follow up with some flags to signify device features
        if (m_dev_prop[idev].concurrentManagedAccess)
            {
            s << ", MNG";
            }
        else
            m_concurrent = false;

        s << std::endl;
        }

    // We print this information in rank order
    msg->collectiveNoticeStr(1,s.str());
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
    // check the CUDA driver version
    int driverVersion = 0;
    cudaError_t error = cudaDriverGetVersion(&driverVersion);

    if (error != cudaSuccess)
        {
        msg->notice(1) << string(cudaGetErrorString(error)) << endl;
        return;
        }

    // determine the number of GPUs that CUDA thinks there is
    int dev_count;
    error = cudaGetDeviceCount(&dev_count);
    if (error != cudaSuccess)
        {
        msg->notice(1) << string(cudaGetErrorString(error)) << endl;
        return;
        }

    // initialize variables
    int n_exclusive_gpus = 0;
    m_gpu_available.resize(dev_count);

    // loop through each GPU and check it's properties
    for (int dev = 0; dev < dev_count; dev++)
        {
        // get the device properties
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, dev);
        if (error != cudaSuccess)
            {
            msg->errorAllRanks() << "Error calling cudaGetDeviceProperties()" << endl;
            throw runtime_error("Error initializing execution configuration");
            }

        // start by assuming that the device is available, it will be excluded later if it is not
        m_gpu_available[dev] = true;

        // exclude the device emulation device
        if (prop.major == 9999 && prop.minor == 9999)
            {
            m_gpu_available[dev] = false;
            msg->notice(2) << "GPU id " << dev << " is not available for computation because "
                           << "it is an emulated device" << endl;
            }

        // exclude a GPU if it's compute version is not high enough
        int compoundComputeVer = prop.minor + prop.major * 10;
        if (m_gpu_available[dev] && compoundComputeVer < CUDA_ARCH)
            {
            m_gpu_available[dev] = false;
            msg->notice(2) << "Notice: GPU id " << dev << " is not available for computation because "
                           << "it's compute capability is not high enough" << endl;

            int min_major = CUDA_ARCH/10;
            int min_minor = CUDA_ARCH - min_major*10;

            msg->notice(2) << "This build of hoomd was compiled for a minimum capability of of " << min_major << "."
                           << min_minor << " but the GPU is only " << prop.major << "." << prop.minor << endl;
            }

        // ignore the display gpu if that was requested
        if (m_gpu_available[dev] && ignore_display && prop.kernelExecTimeoutEnabled)
            {
            m_gpu_available[dev] = false;
            msg->notice(2) << "Notice: GPU id " << dev << " is not available for computation because "
                           << "it appears to be attached to a display" << endl;
            }

        // exclude a gpu if it is compute-prohibited
        if (m_gpu_available[dev] && prop.computeMode == cudaComputeModeProhibited)
            {
            m_gpu_available[dev] = false;
            msg->notice(2) << "Notice: GPU id " << dev << " is not available for computation because "
                           << "it is set in the compute-prohibited mode" << endl;
            }

        // count the number of compute-exclusive gpus
        if (m_gpu_available[dev] &&
            (prop.computeMode == cudaComputeModeExclusive || prop.computeMode == cudaComputeModeExclusiveProcess))
            n_exclusive_gpus++;
        }

    std::vector<gpu_elem> gpu_priorities;
    for (int dev = 0; dev < dev_count; dev++)
        {
        if (m_gpu_available[dev])
            {
            cudaDeviceProp prop;
            cudaError_t error = cudaGetDeviceProperties(&prop, dev);
            if (error != cudaSuccess)
                {
                msg->errorAllRanks() << "Error calling cudaGetDeviceProperties()" << endl;
                throw runtime_error("Error initializing execution configuration");
                }

            // calculate a simple priority: prefer the newest GPUs first, then those with more multiprocessors,
            // then subtract a bit if the device is attached to a display
            float priority = float(prop.major*1000000 + prop.minor*10000 + prop.multiProcessorCount);

            if (prop.kernelExecTimeoutEnabled)
                priority -= 0.1f;

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
    \return The count of available GPUs determined by scanGPUs
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

/*! Print out GPU stats if running on the GPU, otherwise determine and print out the CPU stats
*/
void ExecutionConfiguration::setupStats()
    {
    n_cpu = 1;

    #ifdef ENABLE_CUDA
    if (exec_mode == GPU)
        {
        m_dev_prop.resize(m_gpu_id.size());

        for (int idev = m_gpu_id.size()-1; idev >= 0; idev--)
            {
            cudaSetDevice(m_gpu_id[idev]);
            cudaGetDeviceProperties(&m_dev_prop[idev], m_gpu_id[idev]);
            }

        // initialize dev_prop with device properties of first device for now
        dev_prop = m_dev_prop[0];

        printGPUStats();

        // GPU runs only use 1 CPU core
        n_cpu = 1;
        }
    #endif

    if (exec_mode == CPU)
        {
        ostringstream s;

        s << "HOOMD-blue is running on the CPU" << endl;
        msg->collectiveNoticeStr(1,s.str());
        }
    }

void ExecutionConfiguration::multiGPUBarrier() const
    {
    #ifdef ENABLE_CUDA
    if (getNumActiveGPUs() > 1)
        {
        // record the synchronization point on every GPU after the last kernel has finished, count down in reverse
        for (int idev = m_gpu_id.size() - 1; idev >= 0; --idev)
            {
            cudaSetDevice(m_gpu_id[idev]);
            cudaEventRecord(m_events[idev], 0);
            }

        // wait for all those events on all GPUs
        for (int idev_i = m_gpu_id.size()-1; idev_i >= 0; --idev_i)
            {
            cudaSetDevice(m_gpu_id[idev_i]);
            for (int idev_j = 0; idev_j < (int) m_gpu_id.size(); ++idev_j)
                cudaStreamWaitEvent(0, m_events[idev_j], 0);
            }
        }
    #endif
    }

void ExecutionConfiguration::beginMultiGPU() const
    {
    m_in_multigpu_block = true;

    #ifdef ENABLE_CUDA
    // implement a one-to-n barrier
    if (getNumActiveGPUs() > 1)
        {
        // record a syncrhonization point on GPU 0
        cudaEventRecord(m_events[0], 0);

        // wait for that event on all GPUs (except GPU 0, for which we rely on implicit synchronization)
        for (int idev = m_gpu_id.size()-1; idev >= 1; --idev)
            {
            cudaSetDevice(m_gpu_id[idev]);
            cudaStreamWaitEvent(0, m_events[0], 0);
            }

        // set GPU 0
        cudaSetDevice(m_gpu_id[0]);

        if (isCUDAErrorCheckingEnabled())
            {
            cudaError_t err_sync = cudaGetLastError();
            handleCUDAError(err_sync, __FILE__, __LINE__);
            }
        }
    #endif
    }

void ExecutionConfiguration::endMultiGPU() const
    {
    m_in_multigpu_block = false;

    #ifdef ENABLE_CUDA
    // implement an n-to-one barrier
    if (getNumActiveGPUs() > 1)
        {
        // record the synchronization point on every GPU, except GPU 0
        for (int idev = m_gpu_id.size() - 1; idev >= 1; --idev)
            {
            cudaSetDevice(m_gpu_id[idev]);
            cudaEventRecord(m_events[idev], 0);
            }

        // wait for these events on GPU 0
        cudaSetDevice(m_gpu_id[0]);
        for (int idev = m_gpu_id.size()-1; idev >= 1; --idev)
            {
            cudaStreamWaitEvent(0, m_events[idev], 0);
            }

        if (isCUDAErrorCheckingEnabled())
            {
            cudaError_t err_sync = cudaGetLastError();
            handleCUDAError(err_sync, __FILE__, __LINE__);
            }
        }
    #endif
    }

int ExecutionConfiguration::guessLocalRank(bool &found)
    {
    found = false;

    #ifdef ENABLE_MPI
    // single rank simulations emulate the ENABLE_MPI=off behavior

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size == 1)
        {
        found = false;
        return 0;
        }

    std::vector<std::string> env_vars;
    char *env;

    // setup common environment variables containing local rank information
    env_vars.push_back("MV2_COMM_WORLD_LOCAL_RANK");
    env_vars.push_back("OMPI_COMM_WORLD_LOCAL_RANK");
    env_vars.push_back("JSM_NAMESPACE_LOCAL_RANK");

    std::vector<std::string>::iterator it;

    for (it = env_vars.begin(); it != env_vars.end(); it++)
        {
        if ((env = getenv(it->c_str())) != NULL)
            {
            msg->notice(3) << "Found local rank in: " << *it << std::endl;
            found = true;
            return atoi(env);
            }
        }

    // try SLURM_LOCALID
    if (((env = getenv("SLURM_LOCALID"))) != NULL)
        {
        int num_total_ranks = 0;
        int errors = 0;
        int slurm_localid = atoi(env);

        if (slurm_localid == 0)
            errors = 1;

        // some SLURMs set LOCALID to 0 on all ranks, check for this
        MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, m_mpi_config->getHOOMDWorldCommunicator());
        MPI_Comm_size(m_mpi_config->getHOOMDWorldCommunicator(), &num_total_ranks);
        if (errors == num_total_ranks)
            {
            msg->notice(3) << "SLURM_LOCALID is 0 on all ranks, it cannot be used" << std::endl;
            }
        else
            {
            msg->notice(3) << "Found local rank in: SLURM_LOCALID" << std::endl;
            found = true;
            return slurm_localid;
            }
        }

    // fall back on global rank id
    msg->notice(3) << "Using global rank to select GPUs" << std::endl;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    found = true;
    return global_rank;
    #else
    return 0;
    #endif
    }


void export_ExecutionConfiguration(py::module& m)
    {
    py::class_<ExecutionConfiguration, std::shared_ptr<ExecutionConfiguration> > executionconfiguration(m,"ExecutionConfiguration");
    executionconfiguration.def(py::init< ExecutionConfiguration::executionMode, std::vector<int>, bool, bool,
        std::shared_ptr<MPIConfiguration>, std::shared_ptr<Messenger> >())
        .def("getMPIConfig", &ExecutionConfiguration::getMPIConfig)
        .def("isCUDAEnabled", &ExecutionConfiguration::isCUDAEnabled)
        .def("setCUDAErrorChecking", &ExecutionConfiguration::setCUDAErrorChecking)
        .def("getNumActiveGPUs", &ExecutionConfiguration::getNumActiveGPUs)
#ifdef ENABLE_CUDA
        .def("cudaProfileStart", &ExecutionConfiguration::cudaProfileStart)
        .def("cudaProfileStop", &ExecutionConfiguration::cudaProfileStop)
#endif
        .def("getGPUName", &ExecutionConfiguration::getGPUName)
        .def_readonly("n_cpu", &ExecutionConfiguration::n_cpu)
        .def_readonly("msg", &ExecutionConfiguration::msg)
#ifdef ENABLE_CUDA
        .def("getComputeCapability", &ExecutionConfiguration::getComputeCapabilityAsString)
#endif
        .def("getPartition", &ExecutionConfiguration::getPartition)
        .def("getNRanks", &ExecutionConfiguration::getNRanks)
        .def("getRank", &ExecutionConfiguration::getRank)
#ifdef ENABLE_TBB
        .def("setNumThreads", &ExecutionConfiguration::setNumThreads)
#endif
        .def("getNumThreads", &ExecutionConfiguration::getNumThreads)
        .def("setMemoryTracing", &ExecutionConfiguration::setMemoryTracing)
        .def("getMemoryTracer", &ExecutionConfiguration::getMemoryTracer);
    ;

    py::enum_<ExecutionConfiguration::executionMode>(executionconfiguration,"executionMode")
        .value("GPU", ExecutionConfiguration::executionMode::GPU)
        .value("CPU", ExecutionConfiguration::executionMode::CPU)
        .value("AUTO", ExecutionConfiguration::executionMode::AUTO)
        .export_values()
    ;
    }
