#pragma once

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
class GPUEvalFactory
    {
    public:
        //! Constructor
        GPUEvalFactory(std::shared_ptr<ExecutionConfiguration> exec_conf,
                       const std::string& code,
                       const std::string& include_path,
                       const std::string& include_path_source,
                       const std::string& cuda_devrt_library_path,
                       unsigned int compute_arch)
            : m_exec_conf(exec_conf)
            {
            m_device_ptr.resize(m_exec_conf->getNumActiveGPUs());
            m_alpha_iso_device_ptr.resize(m_exec_conf->getNumActiveGPUs());
            m_alpha_union_device_ptr.resize(m_exec_conf->getNumActiveGPUs());

            compileGPU(code, include_path, include_path_source, cuda_devrt_library_path, compute_arch);
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

        //! Return the device function pointer for a GPU
        /* \param idev the logical GPU id
         */
        eval_func getDeviceFunc(unsigned int idev) const
            {
            assert(m_device_ptr.size() > idev);
            return m_device_ptr[idev];
            }

        void setAlphaPtr(float *d_alpha)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            // copy pointer in device variables
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

    private:
        std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The exceuction configuration
        std::vector<eval_func> m_device_ptr;                //!< The pointer to the device function, for every device

        #ifdef __HIP_PLATFORM_NVCC__
        std::vector<CUdeviceptr> m_alpha_iso_device_ptr;    //!< Device pointer to data ptr for patches
        std::vector<CUdeviceptr> m_alpha_union_device_ptr;  //!< Device pointer to data ptr for union patches
        #endif

        //! Helper function for RTC
        void compileGPU(const std::string& code,
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
