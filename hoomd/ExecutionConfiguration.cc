// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "ExecutionConfiguration.h"
#include "HOOMDVersion.h"

// support converting python lists as arguments to STL vectors
#include <hoomd/extern/pybind/include/pybind11/stl.h>

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

using namespace std;

#ifdef ENABLE_CUDA
#include "CachedAllocator.h"
#endif

/*! \file ExecutionConfiguration.cc
    \brief Defines ExecutionConfiguration and related classes
*/

/*! \param mode Execution mode to set (cpu or gpu)
    \param gpu_id List of GPU IDs on which to run, or empty for automatic selection
    \param min_cpu If set to true, cudaDeviceBlockingSync is set to keep the CPU usage of HOOMD to a minimum
    \param ignore_display If set to true, try to ignore GPUs attached to the display
    \param _msg Messenger to use for status message printing
    \param n_ranks Number of ranks per partition

    Explicitly force the use of either CPU or GPU execution. If GPU exeuction is selected, then a default GPU choice
    is made by not calling cudaSetDevice.
*/
ExecutionConfiguration::ExecutionConfiguration(executionMode mode,
                                               std::vector<unsigned int> gpu_id,
                                               bool min_cpu,
                                               bool ignore_display,
                                               std::shared_ptr<Messenger> _msg,
                                               unsigned int n_ranks
                                               #ifdef ENABLE_MPI
                                               , MPI_Comm hoomd_world
                                               #endif
                                               )
    : m_cuda_error_checking(false), msg(_msg)
    {
    if (!msg)
        msg = std::shared_ptr<Messenger>(new Messenger());

    ostringstream s;
    for (auto it = gpu_id.begin(); it != gpu_id.end(); ++it)
        {
        s << *it << " ";
        }

    msg->notice(5) << "Constructing ExecutionConfiguration: ( " << s.str() << ") " <<  min_cpu << " " << ignore_display << endl;
    exec_mode = mode;

    m_rank = 0;

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

    // now, exec_mode should be either CPU or GPU - proceed with initialization

    // initialize the GPU if that mode was requested
    if (exec_mode == GPU)
        {
        if (!gpu_id.size() && !m_system_compute_exclusive)
            {
            // if we are not running in compute exclusive mode, use
            // local MPI rank as preferred GPU id
            msg->notice(2) << "This system is not compute exclusive, using local rank to select GPUs" << std::endl;
            gpu_id.push_back((guessLocalRank() % dev_count));
            }

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
#endif

    #ifdef ENABLE_MPI
    m_n_rank = n_ranks;
    m_hoomd_world = hoomd_world;
    splitPartitions(hoomd_world);
    #endif

    setupStats();

    #ifdef ENABLE_CUDA
    if (exec_mode == GPU)
        {
        // select first device by default
        cudaSetDevice(m_gpu_id[0]);

        cudaError_t err_sync = cudaGetLastError();
        handleCUDAError(err_sync, __FILE__, __LINE__);

        // initialize cached allocator, max allocation 0.5*global mem
        m_cached_alloc = new CachedAllocator((unsigned int)(0.5f*(float)dev_prop.totalGlobalMem));
        }
    #endif

    #ifdef ENABLE_MPI
    // ensure that all ranks are on the same execution configuration
    if (getNRanks() > 1)
        {
        executionMode rank0_mode = exec_mode;
        bcast(rank0_mode, 0, getMPICommunicator());

        // ensure that all ranks terminate here
        int errors = 0;
        if (rank0_mode != exec_mode)
            errors = 1;

        MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, getMPICommunicator());

        if (errors != 0)
            {
            msg->error() << "Not all ranks have the same execution context (some are CPU and some are GPU)" << endl;
            throw runtime_error("Error initializing execution configuration");
            }
        }

    if (hoomd_launch_timing && getNRanksGlobal() > 1)
        {
        // compute the number of seconds to get an exec conf
        timeval t;
        gettimeofday(&t, NULL);
        unsigned int conf_time = t.tv_sec - hoomd_launch_time;

        // get the min and max times
        unsigned int start_time_min, start_time_max, mpi_init_time_min, mpi_init_time_max, conf_time_min, conf_time_max;
        MPI_Reduce(&hoomd_start_time, &start_time_min, 1, MPI_UNSIGNED, MPI_MIN, 0, m_hoomd_world);
        MPI_Reduce(&hoomd_start_time, &start_time_max, 1, MPI_UNSIGNED, MPI_MAX, 0, m_hoomd_world);

        MPI_Reduce(&hoomd_mpi_init_time, &mpi_init_time_min, 1, MPI_UNSIGNED, MPI_MIN, 0, m_hoomd_world);
        MPI_Reduce(&hoomd_mpi_init_time, &mpi_init_time_max, 1, MPI_UNSIGNED, MPI_MAX, 0, m_hoomd_world);

        MPI_Reduce(&conf_time, &conf_time_min, 1, MPI_UNSIGNED, MPI_MIN, 0, m_hoomd_world);
        MPI_Reduce(&conf_time, &conf_time_max, 1, MPI_UNSIGNED, MPI_MAX, 0, m_hoomd_world);

        // write them out to a file
        if (getRankGlobal() == 0)
            {
            msg->notice(2) << "start_time:    [" << start_time_min << ", " << start_time_max << "]" << std::endl;
            msg->notice(2) << "mpi_init_time: [" << mpi_init_time_min << ", " << mpi_init_time_max << "]" << std::endl;
            msg->notice(2) << "conf_time:     [" << conf_time_min << ", " << conf_time_max << "]" << std::endl;
            }
        }
    #endif

    #ifdef ENABLE_TBB
    m_num_threads = tbb::task_scheduler_init::default_num_threads();

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

    #if defined(ENABLE_CUDA)
    if (exec_mode == GPU)
        {
        delete m_cached_alloc;

        #ifndef ENABLE_MPI_CUDA
        cudaDeviceReset();
        #endif
        }
    #endif

    #ifdef ENABLE_MPI
    // enable Messenger to gracefully finish any MPI-IO
    msg->unsetMPICommunicator();
    #endif
    }

#ifdef ENABLE_MPI
void ExecutionConfiguration::splitPartitions(MPI_Comm mpi_comm)
    {
    m_mpi_comm = mpi_comm;

    int num_total_ranks;
    MPI_Comm_size(m_mpi_comm, &num_total_ranks);

    unsigned int partition = 0;

    if  (m_n_rank != 0)
        {
        int rank;
        MPI_Comm_rank(m_mpi_comm, &rank);

        if (num_total_ranks % m_n_rank != 0)
            {
            msg->error() << "Invalid setting --nrank" << std::endl;
            throw(runtime_error("Error setting up MPI."));
            }

        partition = rank / m_n_rank;

        // Split the communicator
        MPI_Comm new_comm;
        MPI_Comm_split(m_mpi_comm, partition, rank, &new_comm);

        // update communicator
        m_mpi_comm = new_comm;
        }

    int rank;
    MPI_Comm_rank(m_mpi_comm, &rank);
    m_rank = rank;

    msg->setRank(rank, partition);
    msg->setMPICommunicator(m_mpi_comm);
    }
#endif

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
    \note Silently returns an emtpy string if no GPUs are specified
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

/*! \returns Compute capability of the GPU formated as 210 (for compute 2.1 as an example)
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
        msg->error() << string(cudaGetErrorString(err)) << " before "
                     << file << ":" << line << endl;

        // throw an error exception
        throw(runtime_error("CUDA Error"));
        }
    }

/*! \param gpu_id Index for the GPU to initialize, set to -1 for automatic selection
    \param min_cpu If set to true, the cudaDeviceBlockingSync device flag is set

    \pre scanGPUs has been called

    initializeGPU will loop through the specified list of GPUs, validate that each one is available for CUDA use
    and then setup CUDA to use the given GPU. After initialzeGPU completes, cuda calls can be made by the main
    application.
*/
void ExecutionConfiguration::initializeGPU(int gpu_id, bool min_cpu)
    {
    int capable_count = getNumCapableGPUs();
    if (capable_count == 0)
        {
        msg->error() << "No capable GPUs were found!" << endl;
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
        msg->error() << "The specified GPU id (" << gpu_id << ") is invalid." << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    if (gpu_id >= (int)getNumTotalGPUs())
        {
        msg->error() << "The specified GPU id (" << gpu_id << ") is not present in the system." << endl;
        msg->error() << "CUDA reports only " << getNumTotalGPUs() << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    if (!isGPUAvailable(gpu_id))
        {
        msg->error() << "The specified GPU id (" << gpu_id << ") is not available for executing HOOMD." << endl;
        throw runtime_error("Error initializing execution configuration");
        }

    cudaSetDeviceFlags(flags | cudaDeviceMapHost);
    cudaSetValidDevices(&m_gpu_list[0], (int)m_gpu_list.size());

    if (gpu_id != -1)
        {
        cudaSetDevice(gpu_id);
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
    cudaDriverGetVersion(&driverVersion);

    // first handle the situation where no driver is installed (or it is a CUDA 2.1 or earlier driver)
    if (driverVersion == 0)
        {
        msg->notice(2) << "NVIDIA driver not installed or is too old, ignoring any GPUs in the system." << endl;
        return;
        }

    // next, check to see if the driver is capable of running the version of CUDART that HOOMD was compiled against
    if (driverVersion < CUDART_VERSION)
        {
        int driver_major = driverVersion / 1000;
        int driver_minor = (driverVersion - driver_major * 1000) / 10;
        int cudart_major = CUDART_VERSION / 1000;
        int cudart_minor = (CUDART_VERSION - cudart_major * 1000) / 10;

        msg->notice(2) << "The NVIDIA driver only supports CUDA versions up to " << driver_major << "."
             << driver_minor << ", but HOOMD was built against CUDA " << cudart_major << "." << cudart_minor << endl;
        msg->notice(2) << "Ignoring any GPUs in the system." << endl;
        return;
        }

    // determine the number of GPUs that CUDA thinks there is
    int dev_count;
    cudaError_t error = cudaGetDeviceCount(&dev_count);
    if (error != cudaSuccess)
        {
        msg->notice(2) << "Error calling cudaGetDeviceCount(). No NVIDIA driver is present, or this user" << endl;
        msg->notice(2) << "does not have readwrite permissions on /dev/nvidia*" << endl;
        msg->notice(2) << "Ignoring any GPUs in the system." << endl;
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
            msg->error() << "Error calling cudaGetDeviceProperties()" << endl;
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
                msg->error() << "Error calling cudaGetDeviceProperties()" << endl;
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

int ExecutionConfiguration::guessLocalRank()
    {
    #ifdef ENABLE_MPI
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
            msg->notice(3) << "Found local rank in " << *it << std::endl;
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
        MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, m_hoomd_world);
        MPI_Comm_size(m_hoomd_world, &num_total_ranks);
        if (errors == num_total_ranks)
            {
            msg->notice(3) << "SLURM_LOCALID is 0 on all ranks" << std::endl;
            }
        else
            {
            return slurm_localid;
            }
        }

    // fall back on global rank id
    msg->notice(2) << "Unable to identify node local rank information" << std::endl;
    msg->notice(2) << "Using global rank to select GPUs" << std::endl;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    return global_rank;
    #else
    return 0;
    #endif
    }

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

#ifdef ENABLE_MPI
unsigned int ExecutionConfiguration::getNRanks() const
    {
    int size;
    MPI_Comm_size(m_mpi_comm, &size);
    return size;
    }
#endif

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

void export_ExecutionConfiguration(py::module& m)
    {
    py::class_<ExecutionConfiguration, std::shared_ptr<ExecutionConfiguration> > executionconfiguration(m,"ExecutionConfiguration");
    executionconfiguration.def(py::init< ExecutionConfiguration::executionMode, std::vector<unsigned int>, bool, bool, std::shared_ptr<Messenger>, unsigned int >())
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
#ifdef ENABLE_MPI
        .def("getPartition", &ExecutionConfiguration::getPartition)
        .def("getNRanks", &ExecutionConfiguration::getNRanks)
        .def("getRank", &ExecutionConfiguration::getRank)
        .def("guessLocalRank", &ExecutionConfiguration::guessLocalRank)
        .def("barrier", &ExecutionConfiguration::barrier)
        .def_static("getNRanksGlobal", &ExecutionConfiguration::getNRanksGlobal)
        .def_static("getRankGlobal", &ExecutionConfiguration::getRankGlobal)
#endif
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
