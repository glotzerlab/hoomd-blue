#ifdef ENABLE_HIP

#include "hoomd/ExecutionConfiguration.h"

#include <stdio.h>
#include <string>
#include <vector>

#include "GPUEvalFactory.h"

// pybind11 vector bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <hip/hip_runtime.h>

#if __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <nvrtc.h>

void GPUEvalFactory::compileGPU(
    const std::string& code,
    const std::string& kernel_name,
    const std::vector<std::string>& options,
    const std::string& cuda_devrt_library_path,
    const unsigned int compute_arch)
    {
    std::vector<std::string> compile_options = {
        "--gpu-architecture=compute_"+std::to_string(compute_arch),
        "--relocatable-device-code=true",
        "--std=c++11",
#ifdef ENABLE_HPMC_MIXED_PRECISION
        "-DENABLE_HPMC_MIXED_PRECISION",
#endif
        "-DHOOMD_LLVMJIT_BUILD",
        "-D__HIPCC__",
        "-D__HIP_DEVICE_COMPILE__",
        "-D__HIP_PLATFORM_NVCC__",
        };

    for (auto p: options)
        compile_options.push_back(p);

    char *compileParams[compile_options.size()];
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        compileParams[i] = reinterpret_cast<char *>(
            malloc(sizeof(char) * (compile_options[i].length() + 1)));
        snprintf(compileParams[i], compile_options[i].length() + 1, "%s",
                 compile_options[i].c_str());
        }

    m_exec_conf->msg->notice(5) << code << std::endl;

    // compile on each GPU, substituting common headers with fake headers
    auto gpu_map = m_exec_conf->getGPUIds();
    m_program.clear();
    for (int idev = m_exec_conf->getNumActiveGPUs()-1; idev >= 0; idev--)
        {
        m_exec_conf->msg->notice(3) << "Compiling nvrtc code on GPU " << idev << std::endl;
        cudaSetDevice(gpu_map[idev]);
        m_program.push_back(m_cache[idev].program(code, 0, compile_options));
        }

    m_exec_conf->msg->notice(3) << "nvrtc options (notice level 5 shows code):" << std::endl;
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        m_exec_conf->msg->notice(3) << " " << compileParams[i] << std::endl;
        }
    }
#endif
#endif
