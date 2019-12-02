#ifndef _PATCH_ENERGY_JIT_GPU_H_
#define _PATCH_ENERGY_JIT_GPU_H_

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#endif

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
class PYBIND11_EXPORT PatchEnergyJITGPU : public PatchEnergyJIT
    {
    public:
        //! Constructor
        PatchEnergyJITGPU(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir, Scalar r_cut,
                       const unsigned int array_size,
                       const std::string& code,
                       const std::string& include_path,
                       const std::string& include_path_source,
                       const std::string& cuda_devrt_library_path,
                       unsigned int compute_arch)
            : PatchEnergyJIT(exec_conf, llvm_ir, r_cut, array_size)
            {
            m_device_ptr.resize(m_exec_conf->getNumActiveGPUs());
            compileGPU(code, include_path, include_path_source, cuda_devrt_library_path, compute_arch);
            }

        virtual ~PatchEnergyJITGPU()
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
                    m_exec_conf->msg->error() << "cuModuleUnload: " << std::string(error);
                    }
                }
           CUresult status = cuLinkDestroy(m_link_state);
           if (status != CUDA_SUCCESS)
                {
                cuGetErrorString(status, const_cast<const char **>(&error));
                m_exec_conf->msg->error() << "cuLinkDestroy: "<< std::string(error);
                }
            #endif
            }

        //! Return the device function pointer for a GPU
        /* \param idev the logical GPU id
         */
        virtual eval_func getDeviceFunc(unsigned int idev) const
            {
            assert(m_device_ptr.size() > idev);
            return m_device_ptr[idev];
            }

    protected:
        std::vector<eval_func> m_device_ptr;                //!< The pointer to the device function, for every device

    private:
        //! Helper function for RTC
        void compileGPU(const std::string& code,
            const std::string& include_path,
            const std::string& include_path_source,
            const std::string& cuda_devrt_library_path,
            unsigned int compute_arch);

        #ifdef __HIP_PLATFORM_NVCC__
        CUlinkState m_link_state;                           //!< CUDA linker
        std::vector<CUmodule> m_module;                     //!< CUDA module
        #endif
    };

//! Exports the PatchEnergyJIT class to python
inline void export_PatchEnergyJITGPU(pybind11::module &m)
    {
    pybind11::class_<PatchEnergyJITGPU, PatchEnergyJIT, std::shared_ptr<PatchEnergyJITGPU> >(m, "PatchEnergyJITGPU")
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, const std::string&, const std::string&, const std::string&,
                                 unsigned int >())
            ;
    }
#endif
#endif // _PATCH_ENERGY_JIT_GPU_H_
