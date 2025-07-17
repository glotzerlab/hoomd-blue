// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ExecutionConfiguration.h"
#include "HOOMDVersion.h"

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_NVCC__)
#include <cuda_runtime.h>
#endif
#endif

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace std;

#if defined(ENABLE_HIP)
#include "CachedAllocator.h"
#endif

/*! \file ExecutionConfiguration.cc
    \brief Defines ExecutionConfiguration and related classes
*/

namespace hoomd
    {
// initialize static variables
bool ExecutionConfiguration::s_gpu_scan_complete = false;
std::vector<std::string> ExecutionConfiguration::s_gpu_scan_messages;
std::vector<int> ExecutionConfiguration::s_capable_gpu_ids;
std::vector<std::string> ExecutionConfiguration::s_capable_gpu_descriptions;

/*! \param mode Execution mode to set (cpu or gpu)
    \param gpu_id GPU ID to select, or -1 to autoselect.
    \param mpi_config MPI configuration object
    \param _msg Messenger to use for status message printing

    Explicitly force the use of either CPU or GPU execution. If GPU execution is selected, then a
   default GPU choice is made by not calling hipSetDevice.
*/
ExecutionConfiguration::ExecutionConfiguration(executionMode mode,
                                               int gpu_id,
                                               std::shared_ptr<MPIConfiguration> mpi_config,
                                               std::shared_ptr<Messenger> _msg)
    : msg(_msg), m_hip_error_checking(false), m_mpi_config(mpi_config)
    {
    if (!m_mpi_config)
        {
        // create mpi config internally
        m_mpi_config = std::shared_ptr<MPIConfiguration>(new MPIConfiguration());
        }

    if (!msg)
        {
        // create Messenger internally
        msg = std::shared_ptr<Messenger>(new Messenger(m_mpi_config));
        }

    msg->notice(5) << "Constructing ExecutionConfiguration: ( " << gpu_id << ") " << endl;
    exec_mode = mode;

#if defined(ENABLE_HIP)
    // scan the available GPUs
    scanGPUs();
    unsigned int dev_count = (unsigned int)s_capable_gpu_ids.size();

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
        bool using_mpi = false;
#ifdef ENABLE_MPI
        // single rank simulations emulate the ENABLE_MPI=off behavior
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1)
            {
            using_mpi = true;
            }
#endif

        if (gpu_id == -1 && using_mpi)
            {
            int local_rank = guessLocalRank();
            // if we found a local rank, use that to select the GPU
            gpu_id = local_rank % dev_count;

            ostringstream s;
            s << "Selected GPU " << gpu_id << " by MPI rank (" << dev_count << " available)."
              << endl;
            msg->collectiveNoticeStr(4, s.str());
            }

        if (gpu_id == -1)
            {
            // auto-detect a single GPU
            msg->collectiveNoticeStr(4, "Asking the driver to choose a GPU.\n");
            initializeGPU(-1);
            }
        else
            {
            initializeGPU(gpu_id);
            }
        }
#else
    if (exec_mode == GPU)
        {
        throw runtime_error("This build of HOOMD does not include GPU support.");
        }

    exec_mode = CPU;
#endif

    setupStats();

    ostringstream s;
    s << "Device is running on ";
    s << m_active_device_description << endl;
    msg->collectiveNoticeStr(3, s.str());

#if defined(ENABLE_HIP)
    if (exec_mode == GPU)
        {
        // Activate the GPU.
        setDevice();
        // initialize cached allocator, max allocation 0.5*global mem
        m_cached_alloc.reset(
            new CachedAllocator(false, (unsigned int)(0.5f * (float)dev_prop.totalGlobalMem)));
        m_cached_alloc_managed.reset(
            new CachedAllocator(true, (unsigned int)(0.5f * (float)dev_prop.totalGlobalMem)));
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
            throw runtime_error("Ranks have different execution configurations.");
            }
        }
#endif
    }

ExecutionConfiguration::~ExecutionConfiguration()
    {
    msg->notice(5) << "Destroying ExecutionConfiguration" << endl;

#if defined(ENABLE_HIP)
    // the destructors of these objects can issue hip calls, so free them before the device reset
    m_cached_alloc.reset();
    m_cached_alloc_managed.reset();
#endif
    }

#if defined(ENABLE_HIP)

std::pair<unsigned int, unsigned int> ExecutionConfiguration::getComputeCapability() const
    {
    auto result = std::make_pair(0, 0);

    if (exec_mode == GPU)
        {
        result = std::make_pair(m_dev_prop.major, m_dev_prop.minor);
        }

    return result;
    }

void ExecutionConfiguration::handleHIPError(hipError_t err,
                                            const char* file,
                                            unsigned int line) const
    {
    // if there was an error
    if (err != hipSuccess)
        {
        // remove HOOMD_SOURCE_DIR from the front of the file
        if (strlen(file) > strlen(HOOMD_SOURCE_DIR))
            file += strlen(HOOMD_SOURCE_DIR);

        std::ostringstream s;
#ifdef __HIP_PLATFORM_NVCC__
        cudaError_t cuda_error = cudaPeekAtLastError();
        s << "CUDA Error: " << string(cudaGetErrorString(cuda_error));
#else
        s << "HIP Error: " << string(hipGetErrorString(err));
#endif
        s << " before " << file << ":" << line;

        // throw an error exception
        throw(runtime_error(s.str()));
        }
    }

/*! \param gpu_id Index for the GPU to initialize, set to -1 for automatic selection

    initializeGPU will loop through the specified list of GPUs, validate that each one is available
   for CUDA use and then setup CUDA to use the given GPU. After initializeGPU completes, hip calls
   can be made by the main application.
*/
void ExecutionConfiguration::initializeGPU(int gpu_id)
    {
    int capable_count = (int)s_capable_gpu_ids.size();
    if (capable_count == 0)
        {
        std::ostringstream s;
        s << "No supported GPUs are present on this system." << std::endl;
        for (const auto& msg : s_gpu_scan_messages)
            s << msg << std::endl;
        throw runtime_error(s.str());
        }

    if (gpu_id < -1)
        {
        std::ostringstream s;
        s << "Invalid device ID " << gpu_id << " (Use -1 to autoselect a GPU).";
        throw runtime_error(s.str());
        }

    if (gpu_id >= (int)s_capable_gpu_ids.size())
        {
        std::ostringstream s;
        s << "Invalid device ID " << gpu_id << " - select a valid device from:" << std::endl;
        for (const auto& desc : s_capable_gpu_descriptions)
            s << desc << std::endl;
        for (const auto& msg : s_gpu_scan_messages)
            s << msg << std::endl;
        throw runtime_error(s.str());
        }

    if (gpu_id != -1)
        {
#ifdef __HIP_PLATFORM_NVCC__
        cudaSetValidDevices(&s_capable_gpu_ids[gpu_id], 1);
#endif
        hipSetDeviceFlags(hipDeviceMapHost);
        hipSetDevice(s_capable_gpu_ids[gpu_id]);
        }
    else
        {
            // initialize the default CUDA context from one of the capable GPUs
#ifdef __HIP_PLATFORM_NVCC__
        cudaSetValidDevices(&s_capable_gpu_ids[0], (int)s_capable_gpu_ids.size());
#endif
        hipSetDeviceFlags(hipDeviceMapHost);
        hipFree(0);
        }

    int hip_gpu_id;
    hipGetDevice(&hip_gpu_id);

    // record the device
    m_gpu_id = hip_gpu_id;

    hipError_t err_sync = hipPeekAtLastError();
    handleHIPError(err_sync, __FILE__, __LINE__);
    }

std::string ExecutionConfiguration::describeGPU(int id, hipDeviceProp_t prop)
    {
    ostringstream s;
    s << "[" << id << "]";
    s << setw(22) << prop.name;

    // then print the SM count and version
    s << setw(4) << prop.multiProcessorCount << " SM_" << prop.major << "." << prop.minor;

    // and the clock rate
    double ghz = double(prop.clockRate) / 1e6;
    s.precision(3);
    s.fill('0');
    s << " @ " << setw(4) << ghz << " GHz";
    s.fill(' ');

    // and the total amount of memory
    int mib = int(float(prop.totalGlobalMem) / float(1024 * 1024));
    s << ", " << setw(4) << mib << " MiB DRAM";
    return s.str();
    }

void ExecutionConfiguration::scanGPUs()
    {
    if (s_gpu_scan_complete)
        {
        // the scan has already been completed
        return;
        }

    s_gpu_scan_complete = true;

    // determine the number of GPUs that CUDA thinks there is
    int dev_count;
    hipError_t error = hipGetDeviceCount(&dev_count);
    if (error != hipSuccess)
        {
        std::string message = "Failed to get GPU device count: ";
#ifdef __HIP_PLATFORM_NVCC__
        cudaError_t cuda_error = cudaPeekAtLastError();
        message += string(cudaGetErrorString(cuda_error));
#else
        message += string(hipGetErrorString(error));
#endif
        s_gpu_scan_messages.push_back(message);
        return;
        }

    if (dev_count == 0)
        {
        s_gpu_scan_messages.push_back("The GPU runtime reports there are 0 devices.");
        }

    // loop through each GPU and check it's properties
    for (int dev = 0; dev < dev_count; dev++)
        {
        // get the device properties
        hipDeviceProp_t prop;
        hipError_t error = hipGetDeviceProperties(&prop, dev);

        if (error != hipSuccess)
            {
            std::string message = "Failed to get device properties: ";
#ifdef __HIP_PLATFORM_NVCC__
            cudaError_t cuda_error = cudaPeekAtLastError();
            message += string(cudaGetErrorString(cuda_error));
#else
            message += string(hipGetErrorString(error));
#endif
            s_gpu_scan_messages.push_back(message);
            continue;
            }

#ifdef __HIP_PLATFORM_NVCC__
        // exclude a GPU if it's compute version is not high enough
        int compoundComputeVer = prop.minor + prop.major * 10;

        if (compoundComputeVer < CUDA_ARCH)
            {
            ostringstream s;
            s << "The device " << prop.name << " with compute capability " << prop.major << "."
              << prop.minor << " does not support HOOMD-blue.";
            s_gpu_scan_messages.push_back(s.str());
            continue;
            }
#endif

        // exclude a gpu if it is compute-prohibited
        if (prop.computeMode == hipComputeModeProhibited)
            {
            ostringstream s;
            s << "The device " << prop.name << " is in a compute prohibited mode.";
            s_gpu_scan_messages.push_back(s.str());
            continue;
            }

        // exclude a GPU when it doesn't support mapped memory
#ifdef __HIP_PLATFORM_NVCC__
        int supports_managed_memory = 0;
        cudaError_t cuda_error
            = cudaDeviceGetAttribute(&supports_managed_memory, cudaDevAttrManagedMemory, dev);
        if (cuda_error != cudaSuccess)
            {
            s_gpu_scan_messages.push_back("Failed to get device attribute: "
                                          + string(cudaGetErrorString(cuda_error)));
            continue;
            }
        if (!supports_managed_memory)
            {
            ostringstream s;
            s << "The device " << prop.name << " does not support managed memory.";
            s_gpu_scan_messages.push_back(s.str());
            continue;
            }
#endif

        s_capable_gpu_descriptions.push_back(describeGPU((int)s_capable_gpu_ids.size(), prop));
        s_capable_gpu_ids.push_back(dev);
        }
    }

#endif

/*! Print out GPU stats if running on the GPU, otherwise determine and print out the CPU stats
 */
void ExecutionConfiguration::setupStats()
    {
#if defined(ENABLE_HIP)
    if (exec_mode == GPU)
        {
        hipSetDevice(m_gpu_id);
        hipGetDeviceProperties(&m_dev_prop, m_gpu_id);

        m_active_device_description = describeGPU(m_gpu_id, m_dev_prop);

        // initialize dev_prop with device properties of first device for now
        dev_prop = m_dev_prop;
        }
#endif

    if (exec_mode == CPU)
        {
        m_active_device_description = "CPU";
        }
    }

int ExecutionConfiguration::guessLocalRank()
    {
#ifdef ENABLE_MPI
    std::vector<std::string> env_vars;
    char* env_value;

    // setup common environment variables containing local rank information
    env_vars.push_back("MV2_COMM_WORLD_LOCAL_RANK");
    env_vars.push_back("OMPI_COMM_WORLD_LOCAL_RANK");
    env_vars.push_back("JSM_NAMESPACE_LOCAL_RANK");

    // Always check SLURM_LOCALID last to allow other mpi launchers to override.
    env_vars.push_back("SLURM_LOCALID");

    for (const auto& env_var : env_vars)
        {
        if ((env_value = getenv(env_var.c_str())) != NULL)
            {
            int rank = atoi(env_value);
            msg->notice(3) << "Found local rank " << rank << " in: " << env_var << std::endl;
            return rank;
            }
        }

    // fall back on global rank id
    msg->notice(3) << "Using global rank to select GPUs" << std::endl;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    return global_rank;
#else
    return 0;
#endif
    }

namespace detail
    {
void export_ExecutionConfiguration(pybind11::module& m)
    {
    pybind11::class_<ExecutionConfiguration, std::shared_ptr<ExecutionConfiguration>>
        executionconfiguration(m, "ExecutionConfiguration");
    executionconfiguration
        .def(pybind11::init<ExecutionConfiguration::executionMode,
                            int,
                            std::shared_ptr<MPIConfiguration>,
                            std::shared_ptr<Messenger>>())
        .def("getMPIConfig", &ExecutionConfiguration::getMPIConfig)
        .def("isCUDAEnabled", &ExecutionConfiguration::isCUDAEnabled)
        .def("setCUDAErrorChecking", &ExecutionConfiguration::setCUDAErrorChecking)
        .def("isCUDAErrorCheckingEnabled", &ExecutionConfiguration::isCUDAErrorCheckingEnabled)
        .def_readonly("msg", &ExecutionConfiguration::msg)
#if defined(ENABLE_HIP)
        .def("getComputeCapability", &ExecutionConfiguration::getComputeCapability)
        .def("hipProfileStart", &ExecutionConfiguration::hipProfileStart)
        .def("hipProfileStop", &ExecutionConfiguration::hipProfileStop)
#endif
        .def("getPartition", &ExecutionConfiguration::getPartition)
        .def("getNRanks", &ExecutionConfiguration::getNRanks)
        .def("getRank", &ExecutionConfiguration::getRank)
        .def("setMemoryTracing", &ExecutionConfiguration::setMemoryTracing)
        .def("memoryTracingEnabled", &ExecutionConfiguration::memoryTracingEnabled)
        .def_static("getCapableDevices", &ExecutionConfiguration::getCapableDevices)
        .def_static("getScanMessages", &ExecutionConfiguration::getScanMessages)
        .def("getActiveDevice", &ExecutionConfiguration::getActiveDevice);

    pybind11::enum_<ExecutionConfiguration::executionMode>(executionconfiguration, "executionMode")
        .value("GPU", ExecutionConfiguration::executionMode::GPU)
        .value("CPU", ExecutionConfiguration::executionMode::CPU)
        .value("AUTO", ExecutionConfiguration::executionMode::AUTO)
        .export_values();
    }
    } // end namespace detail

    } // end namespace hoomd
