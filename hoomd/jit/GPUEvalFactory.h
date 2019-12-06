#pragma once

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"
#include "EvaluatorUnionGPU.cuh"

#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
/*! This class encapsulates a JIT compiled kernel and provides the API necessary to query kernel
    parameters and launch the kernel into a stream.

    Additionally, it allows access to pointers alpha_iso and alpha_union defined at global scope.
 */
class GPUEvalFactory
    {
    public:
        //! Constructor
        GPUEvalFactory(std::shared_ptr<ExecutionConfiguration> exec_conf,
                       const std::string& code,
                       const std::string& kernel_name,
                       const std::string& include_path,
                       const std::string& include_path_source,
                       const std::string& cuda_devrt_library_path,
                       unsigned int compute_arch)
            : m_exec_conf(exec_conf)
            {
            m_kernel_ptr.resize(m_exec_conf->getNumActiveGPUs());
            m_alpha_iso_device_ptr.resize(m_exec_conf->getNumActiveGPUs());
            m_alpha_union_device_ptr.resize(m_exec_conf->getNumActiveGPUs());

            m_rcut_union_device_ptr.resize(m_exec_conf->getNumActiveGPUs());
            m_union_params_device_ptr.resize(m_exec_conf->getNumActiveGPUs());

            compileGPU(code, kernel_name, include_path, include_path_source, cuda_devrt_library_path, compute_arch);
            }

        ~GPUEvalFactory()
            {
            #ifdef __HIP_PLATFORM_NVCC__
            // free resources
            char *error;
            for (auto m: m_module)
                {
                CUresult status = cuModuleUnload(m);
                if (status != CUDA_SUCCESS)
                    {
                    cuGetErrorString(status, const_cast<const char **>(&error));
                    m_exec_conf->msg->error() << "cuModuleUnload: " << std::string(error) << std::endl;
                    }
                }
           CUresult status = cuLinkDestroy(m_link_state);
           if (status != CUDA_SUCCESS)
                {
                cuGetErrorString(status, const_cast<const char **>(&error));
                m_exec_conf->msg->error() << "cuLinkDestroy: "<< std::string(error) << std::endl;
                }

            nvrtcResult nvrtc_status = nvrtcDestroyProgram(&m_program);
            if (nvrtc_status != NVRTC_SUCCESS)
                m_exec_conf->msg->error() << "nvrtcDestroyProgram: "  << std::string(nvrtcGetErrorString(nvrtc_status)) << std::endl;;
            #endif
            }

        //! Return the __global__ function pointer for a GPU
        /* \param idev the logical GPU id
         */
        const void *getKernelAddress(unsigned int idev) const
            {
            assert(m_kernel_ptr.size() > idev);
            return reinterpret_cast<const void *>(m_kernel_ptr[idev]);
            }

        //! Return the maximum number of threads per block for this kernel
        /* \param idev the logical GPU id
         */
        unsigned int getKernelMaxThreads(unsigned int idev) const
            {
            assert(m_kernel_ptr.size() > idev);
            int max_threads = 0;
            CUresult custatus = cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_kernel_ptr[idev]);
            char *error;
            if (custatus != CUDA_SUCCESS)
                {
                cuGetErrorString(custatus, const_cast<const char **>(&error));
                throw std::runtime_error("cufuncGetAttribute: "+std::string(error));
                }
            return max_threads;
            }

        //! Return the shared size usage in bytes for this kernel
        /* \param idev the logical GPU id
         */
        unsigned int getKernelSharedSize(unsigned int idev) const
            {
            assert(m_kernel_ptr.size() > idev);
            int shared_bytes = 0;
            CUresult custatus = cuFuncGetAttribute(&shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, m_kernel_ptr[idev]);
            char *error;
            if (custatus != CUDA_SUCCESS)
                {
                cuGetErrorString(custatus, const_cast<const char **>(&error));
                throw std::runtime_error("cufuncGetAttribute: "+std::string(error));
                }
            return shared_bytes;
            }

        //! Asynchronously launch the JIT kernel
        /*! \param idev logical GPU id to launch on
            \param grid The grid dimensions
            \param threads The thread block dimensions
            \param sharedMemBytes The size of the dynamic shared mem allocation
            \param hStream stream to execute on
            \param kernelParams the kernel parameters
            */
        void launchKernel(unsigned int idev, dim3 grid, dim3 threads, unsigned int sharedMemBytes, hipStream_t hStream, void** kernelParams) const
            {
            CUresult custatus = cuLaunchKernel(m_kernel_ptr[idev],
                grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, sharedMemBytes, hStream, kernelParams, 0);
            char *error;
            if (custatus != CUDA_SUCCESS)
                {
                cuGetErrorString(custatus, const_cast<const char **>(&error));
                throw std::runtime_error("cuLaunchKernel: "+std::string(error));
                }
            }

        void setAlphaPtr(float *d_alpha)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            // copy pointer to device variables
            auto gpu_map = m_exec_conf->getGPUIds();
            for (int idev = m_exec_conf->getNumActiveGPUs()-1; idev >= 0; --idev)
                {
                cudaSetDevice(gpu_map[idev]);

                // copy the array pointer to the device
                char *error;
                CUresult custatus = cuMemcpyHtoD(m_alpha_iso_device_ptr[idev], &d_alpha, sizeof(float *));
                if (custatus != CUDA_SUCCESS)
                    {
                    cuGetErrorString(custatus, const_cast<const char **>(&error));
                    throw std::runtime_error("cuMemcpyHtoD: "+std::string(error));
                    }
                }
            #endif
            }

        void setAlphaUnionPtr(float *d_alpha_union)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            // copy pointer to device variables
            auto gpu_map = m_exec_conf->getGPUIds();
            for (int idev = m_exec_conf->getNumActiveGPUs()-1; idev >= 0; --idev)
                {
                cudaSetDevice(gpu_map[idev]);

                // copy the array pointer to the device
                char *error;
                CUresult custatus = cuMemcpyHtoD(m_alpha_union_device_ptr[idev], &d_alpha_union, sizeof(float *));
                if (custatus != CUDA_SUCCESS)
                    {
                    cuGetErrorString(custatus, const_cast<const char **>(&error));
                    throw std::runtime_error("cuMemcpyHtoD: "+std::string(error));
                    }
                }
            #endif
            }

        void setRCutUnion(float rcut)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            // copy pointer to device variable
            auto gpu_map = m_exec_conf->getGPUIds();
            for (int idev = m_exec_conf->getNumActiveGPUs()-1; idev >= 0; --idev)
                {
                cudaSetDevice(gpu_map[idev]);

                // copy the array pointer to the device
                char *error;
                CUresult custatus = cuMemcpyHtoD(m_rcut_union_device_ptr[idev], &rcut, sizeof(float));
                if (custatus != CUDA_SUCCESS)
                    {
                    cuGetErrorString(custatus, const_cast<const char **>(&error));
                    throw std::runtime_error("cuMemcpyHtoD: "+std::string(error));
                    }
                }
            #endif
            }

        void setUnionParamsPtr(jit::union_params_t *d_params)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            // copy pointer to device variables
            auto gpu_map = m_exec_conf->getGPUIds();
            for (int idev = m_exec_conf->getNumActiveGPUs()-1; idev >= 0; --idev)
                {
                cudaSetDevice(gpu_map[idev]);

                // copy the array pointer to the device
                char *error;
                CUresult custatus = cuMemcpyHtoD(m_union_params_device_ptr[idev], &d_params, sizeof(jit::union_params_t *));
                if (custatus != CUDA_SUCCESS)
                    {
                    cuGetErrorString(custatus, const_cast<const char **>(&error));
                    throw std::runtime_error("cuMemcpyHtoD: "+std::string(error));
                    }
                }
            #endif
            }

    private:
        std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The exceuction configuration
        std::vector<CUfunction> m_kernel_ptr;              //!< The pointer to the kernel, for every device

        #ifdef __HIP_PLATFORM_NVCC__
        std::vector<CUdeviceptr> m_alpha_iso_device_ptr;     //!< Device pointer to data ptr for patches
        std::vector<CUdeviceptr> m_alpha_union_device_ptr;   //!< Device pointer to data ptr for union patches
        std::vector<CUdeviceptr> m_rcut_union_device_ptr;    //!< Device pointer to data ptr for union patches
        std::vector<CUdeviceptr> m_union_params_device_ptr;  //!< Device pointer to data ptr for union patches
        #endif

        //! Helper function for RTC
        void compileGPU(const std::string& code,
            const std::string& kernel_name,
            const std::string& include_path,
            const std::string& include_path_source,
            const std::string& cuda_devrt_library_path,
            unsigned int compute_arch);

        #ifdef __HIP_PLATFORM_NVCC__
        CUlinkState m_link_state;                           //!< CUDA linker
        nvrtcProgram m_program;                             //!< NVRTC program
        std::vector<CUmodule> m_module;                     //!< CUDA module
        #endif
    };
#endif
