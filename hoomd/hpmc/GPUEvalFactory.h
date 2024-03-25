// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP

#include "EvaluatorUnionGPU.cuh"
#include "PatchEnergyJIT.h"
#include "hoomd/hpmc/IntegratorHPMC.h"

#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#ifdef ENABLE_DEBUG_JIT
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1
#else
#define JITIFY_PRINT_LOG 0
#define JITIFY_PRINT_LAUNCH 0
#endif
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE 0
#define JITIFY_PRINT_PTX 0
#define JITIFY_PRINT_HEADER_PATHS 0

#undef DEVICE
#include "hoomd/extern/jitify.hpp"

#endif

#include <map>
#include <vector>

namespace hoomd
    {
namespace hpmc
    {
//! Evaluate patch energies via runtime generated code, GPU version
/*! This class encapsulates a JIT compiled kernel and provides the API necessary to query kernel
    parameters and launch the kernel into a stream.

    Additionally, it allows access to pointers param_array and alpha_union
    defined at global scope.
 */
class GPUEvalFactory
    {
    public:
    //! Constructor
    GPUEvalFactory(std::shared_ptr<ExecutionConfiguration> exec_conf,
                   const std::string& code,
                   const std::string& kernel_name,
                   const std::vector<std::string>& options,
                   const std::string& cuda_devrt_library_path,
                   unsigned int compute_arch)
        : m_exec_conf(exec_conf), m_kernel_name(kernel_name)
        {
        for (unsigned int i = 1; i <= (unsigned int)m_exec_conf->dev_prop.warpSize; i *= 2)
            m_eval_threads.push_back(i);

        for (unsigned int i = exec_conf->dev_prop.warpSize;
             i <= (unsigned int)m_exec_conf->dev_prop.maxThreadsPerBlock;
             i *= 2)
            m_launch_bounds.push_back(i);

// instantiate jitify cache
#ifdef __HIP_PLATFORM_NVCC__
        m_cache.resize(this->m_exec_conf->getNumActiveGPUs());
#endif

        compileGPU(code, kernel_name, options, cuda_devrt_library_path, compute_arch);
        }

    ~GPUEvalFactory() { }

    //! Return the list of available launch bounds
    /* \param idev the logical GPU id
       \param eval_threads template parameter
     */
    const std::vector<unsigned int>& getLaunchBounds() const
        {
        return m_launch_bounds;
        }

    //! Return the maximum number of threads per block for this kernel
    /* \param idev the logical GPU id
       \param eval_threads template parameter
       \param launch_bounds template parameter
     */
    unsigned int
    getKernelMaxThreads(unsigned int idev, unsigned int eval_threads, unsigned int launch_bounds)
        {
        int max_threads = 0;

#ifdef __HIP_PLATFORM_NVCC__
        CUresult custatus = cuFuncGetAttribute(
            &max_threads,
            CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            m_program[idev].kernel(m_kernel_name).instantiate(eval_threads, launch_bounds));
        char* error;
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char**>(&error));
            throw std::runtime_error("cuFuncGetAttribute: " + std::string(error));
            }
#endif

        return max_threads;
        }

    //! Return the shared size usage in bytes for this kernel
    /* \param idev the logical GPU id
       \param eval_threads template parameter
       \param launch_bounds template parameter
     */
    unsigned int
    getKernelSharedSize(unsigned int idev, unsigned int eval_threads, unsigned int launch_bounds)
        {
        int shared_size = 0;

#ifdef __HIP_PLATFORM_NVCC__
        CUresult custatus = cuFuncGetAttribute(
            &shared_size,
            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            m_program[idev].kernel(m_kernel_name).instantiate(eval_threads, launch_bounds));
        char* error;
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char**>(&error));
            throw std::runtime_error("cuFuncGetAttribute: " + std::string(error));
            }
#endif

        return shared_size;
        }

//! Asynchronously launch the JIT kernel
/*! \param idev logical GPU id to launch on
    \param grid The grid dimensions
    \param threads The thread block dimensions
    \param sharedMemBytes The size of the dynamic shared mem allocation
    \param hStream stream to execute on
    \param eval_threads template parameter
    \param launch_bounds template parameter
    */
#ifdef __HIP_PLATFORM_NVCC__
    jitify::KernelLauncher configureKernel(unsigned int idev,
                                           dim3 grid,
                                           dim3 threads,
                                           size_t sharedMemBytes,
                                           cudaStream_t hStream,
                                           unsigned int eval_threads,
                                           unsigned int launch_bounds)
        {
        cudaSetDevice(m_exec_conf->getGPUIds()[idev]);

        return m_program[idev]
            .kernel(m_kernel_name)
            .instantiate(eval_threads, launch_bounds)
            .configure(grid, threads, static_cast<unsigned int>(sharedMemBytes), hStream);
        }
#endif

    void setAlphaPtr(float* d_alpha, bool is_union)
        {
#ifdef __HIP_PLATFORM_NVCC__
        auto gpu_map = m_exec_conf->getGPUIds();
        std::string param_array_name = "param_array";
        if (is_union)
            {
            param_array_name += "_isotropic";
            }
        for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            cudaSetDevice(gpu_map[idev]);

            for (auto e : m_eval_threads)
                {
                for (auto l : m_launch_bounds)
                    {
                    CUdeviceptr ptr = m_program[idev]
                                          .kernel(m_kernel_name)
                                          .instantiate(e, l)
                                          .get_global_ptr(param_array_name.c_str());

                    // copy the array pointer to the device
                    char* error;
                    CUresult custatus = cuMemcpyHtoD(ptr, &d_alpha, sizeof(float*));
                    if (custatus != CUDA_SUCCESS)
                        {
                        cuGetErrorString(custatus, const_cast<const char**>(&error));
                        throw std::runtime_error("cuMemcpyHtoD: " + std::string(error));
                        }
                    }
                }
            }
#endif
        }

    void setAlphaUnionPtr(float* d_alpha_union)
        {
#ifdef __HIP_PLATFORM_NVCC__
        auto gpu_map = m_exec_conf->getGPUIds();
        for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            cudaSetDevice(gpu_map[idev]);

            for (auto e : m_eval_threads)
                {
                for (auto l : m_launch_bounds)
                    {
                    CUdeviceptr ptr = m_program[idev]
                                          .kernel(m_kernel_name)
                                          .instantiate(e, l)
                                          .get_global_ptr("param_array_constituent");

                    // copy the array pointer to the device
                    char* error;
                    CUresult custatus = cuMemcpyHtoD(ptr, &d_alpha_union, sizeof(float*));
                    if (custatus != CUDA_SUCCESS)
                        {
                        cuGetErrorString(custatus, const_cast<const char**>(&error));
                        throw std::runtime_error("cuMemcpyHtoD: " + std::string(error));
                        }
                    }
                }
            }
#endif
        }

    void setRCutUnion(float rcut)
        {
#ifdef __HIP_PLATFORM_NVCC__
        auto gpu_map = m_exec_conf->getGPUIds();
        for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            cudaSetDevice(gpu_map[idev]);

            for (auto e : m_eval_threads)
                {
                for (auto l : m_launch_bounds)
                    {
                    CUdeviceptr ptr = m_program[idev]
                                          .kernel(m_kernel_name)
                                          .instantiate(e, l)
                                          .get_global_ptr("hoomd::hpmc::jit::d_r_cut_constituent");

                    // copy the array pointer to the device
                    char* error;
                    CUresult custatus = cuMemcpyHtoD(ptr, &rcut, sizeof(float));
                    if (custatus != CUDA_SUCCESS)
                        {
                        cuGetErrorString(custatus, const_cast<const char**>(&error));
                        throw std::runtime_error("cuMemcpyHtoD: " + std::string(error));
                        }
                    }
                }
            }
#endif
        }

    void setUnionParamsPtr(jit::union_params_t* d_params)
        {
#ifdef __HIP_PLATFORM_NVCC__
        auto gpu_map = m_exec_conf->getGPUIds();
        for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            cudaSetDevice(gpu_map[idev]);

            for (auto e : m_eval_threads)
                {
                for (auto l : m_launch_bounds)
                    {
                    CUdeviceptr ptr = m_program[idev]
                                          .kernel(m_kernel_name)
                                          .instantiate(e, l)
                                          .get_global_ptr("hoomd::hpmc::jit::d_union_params");

                    // copy the array pointer to the device
                    char* error;
                    CUresult custatus = cuMemcpyHtoD(ptr, &d_params, sizeof(jit::union_params_t*));
                    if (custatus != CUDA_SUCCESS)
                        {
                        cuGetErrorString(custatus, const_cast<const char**>(&error));
                        throw std::runtime_error("cuMemcpyHtoD: " + std::string(error));
                        }
                    }
                }
            }
#endif
        }

    private:
    std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The exceuction configuration
    std::vector<unsigned int> m_eval_threads;            //!< The number of template paramteres
    std::vector<unsigned int> m_launch_bounds; //!< The number of different __launch_bounds__
    const std::string m_kernel_name;           //!< The name of the __global__ function

    //! Helper function for RTC
    void compileGPU(const std::string& code,
                    const std::string& kernel_name,
                    const std::vector<std::string>& options,
                    const std::string& cuda_devrt_library_path,
                    unsigned int compute_arch);

#ifdef __HIP_PLATFORM_NVCC__
    std::vector<jitify::JitCache> m_cache;  //!< jitify kernel cache, one per GPU
    std::vector<jitify::Program> m_program; //!< The kernel object, one per GPU
#endif
    };

    } // end namespace hpmc
    } // end namespace hoomd
#endif
