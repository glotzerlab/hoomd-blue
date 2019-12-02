#ifdef ENABLE_HIP

#include "hoomd/ExecutionConfiguration.h"

#include <stdio.h>
#include <string>
#include <vector>

#include "PatchEnergyJITGPU.h"

// pybind11 vector bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#if __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <nvrtc.h>

#include "jitify_safe_headers.hpp"

void PatchEnergyJITGPU::compileGPU(
    const std::string& code,
    const std::string& include_path,
    const std::string& include_path_source,
    const std::string& cuda_devrt_library_path,
    const unsigned int compute_arch)
    {
    std::vector<std::string> compile_options = {
        "--gpu-architecture=compute_"+std::to_string(compute_arch),
        "--include-path="+include_path,
        "--include-path="+include_path_source,
        "--relocatable-device-code=true",
        "--device-as-default-execution-space",
        "-DHOOMD_LLVMJIT_BUILD",
        "-D__HIPCC__"
        };

    char *compileParams[compile_options.size()];
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        compileParams[i] = reinterpret_cast<char *>(
            malloc(sizeof(char) * (compile_options[i].length() + 1)));
        snprintf(compileParams[i], compile_options[i].length() + 1, "%s",
                 compile_options[i].c_str());
        }

    // compile

    // add fake math headers + printf
    std::string printf_include = std::string("#define FILE int\n") +
        std::string("int fflush ( FILE * stream );\n") +
        std::string("int fprintf ( FILE * stream, const char * format, ... );\n");

    std::string code_with_headers = std::string(jitify::detail::jitsafe_header_math) +
        printf_include + code;

    nvrtcProgram prog;
    nvrtcResult status = nvrtcCreateProgram(&prog, code_with_headers.c_str(), "evaluator.cu", 0, NULL, NULL);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcCreateProgram error: "+std::string(nvrtcGetErrorString(status)));

    m_exec_conf->msg->notice(3) << "nvrtc options:" << std::endl;
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        m_exec_conf->msg->notice(3) << " " << compileParams[i] << std::endl;
        }

    char eval_name[] = "&p_eval";
    status = nvrtcAddNameExpression(prog, eval_name);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcAddNameExpression error: "+std::string(nvrtcGetErrorString(status)));

    nvrtcResult compile_result = nvrtcCompileProgram(prog, compile_options.size(), compileParams);
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        free(compileParams[i]);
        }

    // Obtain compilation log from the program.
    size_t logSize;
    status = nvrtcGetProgramLogSize(prog, &logSize);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetProgramLogSize error: "+std::string(nvrtcGetErrorString(status)));

    char *log = new char[logSize];
    status = nvrtcGetProgramLog(prog, log);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetProgramLog error: "+std::string(nvrtcGetErrorString(status)));
    m_exec_conf->msg->notice(3) << "nvrtc output" << std::endl << log << '\n';
    delete[] log;

    if (compile_result != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcCompileProgram error: "+std::string(nvrtcGetErrorString(compile_result)));

    // fetch PTX
    size_t ptx_size;
    status = nvrtcGetPTXSize(prog, &ptx_size);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetPTXSize error: "+std::string(nvrtcGetErrorString(status)));

    char ptx[ptx_size];
    status = nvrtcGetPTX(prog, ptx);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetPTX error: "+std::string(nvrtcGetErrorString(status)));

    // look up mangled name
    char *eval_name_mangled;
    status = nvrtcGetLoweredName(prog, eval_name, const_cast<const char **>(&eval_name_mangled));
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetLoweredNane: "+std::string(nvrtcGetErrorString(status)));

    // link PTX into cubin
    CUresult custatus;
    char *error;
    custatus = cuLinkCreate(0, 0, 0, &m_link_state);
    if (custatus != CUDA_SUCCESS)
        {
        cuGetErrorString(custatus, const_cast<const char **>(&error));
        throw std::runtime_error("cuLinkCreate: "+std::string(error));
        }

    custatus = cuLinkAddFile(m_link_state, CU_JIT_INPUT_LIBRARY, cuda_devrt_library_path.c_str(), 0, 0, 0);
    if (custatus != CUDA_SUCCESS)
        {
        cuGetErrorString(custatus, const_cast<const char **>(&error));
        throw std::runtime_error("cuLinkAddFile: "+std::string(error));
        }

    custatus = cuLinkAddData(m_link_state, CU_JIT_INPUT_PTX, (void *) ptx, ptx_size, "evaluator.ptx", 0, 0, 0);
    if (custatus != CUDA_SUCCESS)
        {
        cuGetErrorString(custatus, const_cast<const char **>(&error));
        throw std::runtime_error("cuLinkAddData: "+std::string(error));
        }

    size_t cubinSize;
    void *cubin;

    custatus = cuLinkComplete(m_link_state, &cubin, &cubinSize);
    if (custatus != CUDA_SUCCESS)
        {
        cuGetErrorString(custatus, const_cast<const char **>(&error));
        throw std::runtime_error("cuLinkComplete: "+std::string(error));
        }

    // load cubin into active contexts, and return function pointers
    auto gpu_map = m_exec_conf->getGPUIds();
    m_module.resize(m_exec_conf->getNumActiveGPUs());
    for (int idev = m_exec_conf->getNumActiveGPUs()-1; idev >= 0; --idev)
        {
        cudaSetDevice(gpu_map[idev]);

        // get module
        custatus = cuModuleLoadData(&m_module[idev], cubin);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleLoadData: "+std::string(error));
            }

        // get variable pointer
        CUdeviceptr p_eval_ptr;
        custatus = cuModuleGetGlobal(&p_eval_ptr, NULL, m_module[idev], eval_name_mangled);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleGetGlobal: "+std::string(error));
            }

        // copy variable contents (the function pointer) to host
        custatus = cuMemcpyDtoH(&m_device_ptr[idev], p_eval_ptr, sizeof(eval_func));
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuMemcpyDtoH: "+std::string(error));
            }
        }

    status = nvrtcDestroyProgram(&prog);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcDestroyProgram: "+std::string(nvrtcGetErrorString(status)));
    }
#endif
#endif
