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

#include "jitify_safe_headers.hpp"

void GPUEvalFactory::compileGPU(
    const std::string& code,
    const std::string& kernel_name,
    const std::string& include_path,
    const std::string& include_path_source,
    const std::string& cuda_devrt_library_path,
    const unsigned int compute_arch)
    {
    m_exec_conf->msg->notice(3) << "Compiling nvrtc code" << std::endl;

    std::vector<std::string> compile_options = {
        "--gpu-architecture=compute_"+std::to_string(compute_arch),
        "--include-path="+include_path,
        "--include-path="+include_path_source,
        "--relocatable-device-code=true",
        "--device-as-default-execution-space",
        "--std=c++11",
#ifdef ENABLE_HPMC_MIXED_PRECISION
        "-DENABLE_HPMC_MIXED_PRECISION",
#endif
        "-DHOOMD_LLVMJIT_BUILD",
        "-D__HIPCC__",
        "-D__HIP_DEVICE_COMPILE__",
        "-D__HIP_PLATFORM_NVCC__",
        "-lineinfo"
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

    m_exec_conf->msg->notice(4) << code_with_headers << std::endl;

    nvrtcResult status = nvrtcCreateProgram(&m_program, code_with_headers.c_str(), "evaluator.cu", 0, NULL, NULL);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcCreateProgram error: "+std::string(nvrtcGetErrorString(status)));

    m_exec_conf->msg->notice(3) << "nvrtc options (notice level 4 shows code):" << std::endl;
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        m_exec_conf->msg->notice(3) << " " << compileParams[i] << std::endl;
        }

    status = nvrtcAddNameExpression(m_program, kernel_name.c_str());
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcAddNameExpression error: "+std::string(nvrtcGetErrorString(status)));

    std::string alpha_iso_name = "&alpha_iso";
    status = nvrtcAddNameExpression(m_program, alpha_iso_name.c_str());
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcAddNameExpression error: "+std::string(nvrtcGetErrorString(status)));

    std::string alpha_union_name = "&alpha_union";
    status = nvrtcAddNameExpression(m_program, alpha_union_name.c_str());
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcAddNameExpression error: "+std::string(nvrtcGetErrorString(status)));

    std::string rcut_name = "&jit::d_rcut_union";
    status = nvrtcAddNameExpression(m_program, rcut_name.c_str());
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcAddNameExpression error: "+std::string(nvrtcGetErrorString(status)));

    std::string union_params_name = "&jit::d_union_params";
    status = nvrtcAddNameExpression(m_program, union_params_name.c_str());
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcAddNameExpression error: "+std::string(nvrtcGetErrorString(status)));

    nvrtcResult compile_result = nvrtcCompileProgram(m_program, compile_options.size(), compileParams);
    for (unsigned int i = 0; i < compile_options.size(); ++i)
        {
        free(compileParams[i]);
        }

    // Obtain compilation log from the program.
    size_t logSize;
    status = nvrtcGetProgramLogSize(m_program, &logSize);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetProgramLogSize error: "+std::string(nvrtcGetErrorString(status)));

    char *log = new char[logSize];
    status = nvrtcGetProgramLog(m_program, log);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetProgramLog error: "+std::string(nvrtcGetErrorString(status)));
    m_exec_conf->msg->notice(3) << "nvrtc output" << std::endl << log << '\n';
    delete[] log;

    if (compile_result != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcCompileProgram error (set device.notice_level=3 to see full log): "+std::string(nvrtcGetErrorString(compile_result)));

    // fetch PTX
    size_t ptx_size;
    status = nvrtcGetPTXSize(m_program, &ptx_size);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetPTXSize error: "+std::string(nvrtcGetErrorString(status)));

    char ptx[ptx_size];
    status = nvrtcGetPTX(m_program, ptx);
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetPTX error: "+std::string(nvrtcGetErrorString(status)));

    // look up mangled names
    char *kernel_name_mangled;
    status = nvrtcGetLoweredName(m_program, kernel_name.c_str(), const_cast<const char **>(&kernel_name_mangled));
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetLoweredName: "+std::string(nvrtcGetErrorString(status)));

    char *alpha_iso_name_mangled;
    status = nvrtcGetLoweredName(m_program, alpha_iso_name.c_str(), const_cast<const char **>(&alpha_iso_name_mangled));
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetLoweredName: "+std::string(nvrtcGetErrorString(status)));

    char *alpha_union_name_mangled;
    status = nvrtcGetLoweredName(m_program, alpha_union_name.c_str(), const_cast<const char **>(&alpha_union_name_mangled));
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetLoweredName: "+std::string(nvrtcGetErrorString(status)));

    char *rcut_name_mangled;
    status = nvrtcGetLoweredName(m_program, rcut_name.c_str(), const_cast<const char **>(&rcut_name_mangled));
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetLoweredName: "+std::string(nvrtcGetErrorString(status)));

    char *union_params_name_mangled;
    status = nvrtcGetLoweredName(m_program, union_params_name.c_str(), const_cast<const char **>(&union_params_name_mangled));
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcGetLoweredName: "+std::string(nvrtcGetErrorString(status)));

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

        // get variable pointers
        CUfunction kernel_ptr;
        custatus = cuModuleGetFunction(&kernel_ptr, m_module[idev], kernel_name_mangled);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleGetFunction: "+std::string(error));
            }
        m_kernel_ptr[idev] = kernel_ptr;

        // get variable pointers
        CUdeviceptr alpha_iso_ptr;
        custatus = cuModuleGetGlobal(&alpha_iso_ptr, 0, m_module[idev], alpha_iso_name_mangled);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleGetGlobal: "+std::string(error));
            }

        m_alpha_iso_device_ptr[idev] = alpha_iso_ptr;

        CUdeviceptr alpha_union_ptr;
        custatus = cuModuleGetGlobal(&alpha_union_ptr, 0, m_module[idev], alpha_union_name_mangled);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleGetGlobal: "+std::string(error));
            }
        m_alpha_union_device_ptr[idev] = alpha_union_ptr;

        CUdeviceptr rcut_ptr;
        custatus = cuModuleGetGlobal(&rcut_ptr, 0, m_module[idev], rcut_name_mangled);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleGetGlobal: "+std::string(error));
            }
        m_rcut_union_device_ptr[idev] = rcut_ptr;

        CUdeviceptr union_params_ptr;
        custatus = cuModuleGetGlobal(&union_params_ptr, 0, m_module[idev], union_params_name_mangled);
        if (custatus != CUDA_SUCCESS)
            {
            cuGetErrorString(custatus, const_cast<const char **>(&error));
            throw std::runtime_error("cuModuleGetGlobal: "+std::string(error));
            }
        m_union_params_device_ptr[idev] = union_params_ptr;
        }
    }
#endif
#endif
