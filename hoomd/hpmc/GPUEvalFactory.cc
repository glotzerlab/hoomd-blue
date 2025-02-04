// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
#endif

namespace hoomd
    {
namespace hpmc
    {
void GPUEvalFactory::compileGPU(const std::string& code,
                                const std::string& kernel_name,
                                const std::vector<std::string>& options,
                                const std::string& cuda_devrt_library_path,
                                const unsigned int compute_arch)
    {
#if __HIP_PLATFORM_NVCC__
#if HOOMD_LONGREAL_SIZE == 32
    std::string longreal_size_str("32");
#else
    std::string longreal_size_str("64");
#endif

#if HOOMD_SHORTREAL_SIZE == 32
    std::string shortreal_size_str("32");
#else
    std::string shortreal_size_str("64");
#endif

    std::vector<std::string> compile_options = {
        "--gpu-architecture=compute_" + std::to_string(compute_arch),
        "--relocatable-device-code=true",
        "--std=c++14",
        "-DHOOMD_LLVMJIT_BUILD",
        "-DHOOMD_LONGREAL_SIZE=" + longreal_size_str,
        "-DHOOMD_SHORTREAL_SIZE=" + shortreal_size_str,
        "-D__HIPCC__",
        "-D__HIP_DEVICE_COMPILE__",
        "-D__HIP_PLATFORM_NVCC__",
    };

    for (auto p : options)
        compile_options.push_back(p);

    char* compileParams[compile_options.size()];
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        compileParams[i]
            = reinterpret_cast<char*>(malloc(sizeof(char) * (compile_options[i].length() + 1)));
        snprintf(compileParams[i],
                 compile_options[i].length() + 1,
                 "%s",
                 compile_options[i].c_str());
        }

    m_exec_conf->msg->notice(3) << "nvrtc options (notice level 5 shows code):" << std::endl;
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        m_exec_conf->msg->notice(3) << " " << compileParams[i] << std::endl;
        }
    m_exec_conf->msg->notice(5) << code << std::endl;

    // compile on each GPU, substituting common headers with fake headers
    auto gpu_map = m_exec_conf->getGPUIds();
    m_program.clear();
    for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; idev--)
        {
        m_exec_conf->msg->notice(3) << "Compiling nvrtc code on GPU " << idev << std::endl;
        cudaSetDevice(gpu_map[idev]);
        m_program.push_back(m_cache[idev].program(code, 0, compile_options));
        }

#endif
    }

    } // end namespace hpmc
    } // end namespace hoomd
#endif
