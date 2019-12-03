#ifndef _PATCH_ENERGY_JIT_GPU_H_
#define _PATCH_ENERGY_JIT_GPU_H_

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"
#include "GPUEvalFactory.h"

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
            : PatchEnergyJIT(exec_conf, llvm_ir, r_cut, array_size),
              m_gpu_factory(exec_conf, code, include_path, include_path_source, cuda_devrt_library_path, compute_arch)
            {
            // allocate data array
            cudaMallocManaged(&m_d_alpha, sizeof(float)*m_alpha_size);
            CHECK_CUDA_ERROR();

            // set the pointer on the device
            m_gpu_factory.setAlphaPtr(m_d_alpha);
            }

        virtual ~PatchEnergyJITGPU()
            {
            cudaFree(m_d_alpha);
            }

        //! Return the device function pointer for a GPU
        /* \param idev the logical GPU id
         */
        virtual eval_func getDeviceFunc(unsigned int idev) const
            {
            return m_gpu_factory.getDeviceFunc(idev);
            }

    protected:
        //! Set the pointer to the auxillary data
        void setAlphaPtr(float *d_alpha)
            {
            // set it in the base class
            PatchEnergyJIT::setAlphaPtr(d_alpha);
            m_gpu_factory.setAlphaPtr(d_alpha);
            }

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation
        float *m_d_alpha;                                   //!< device memory holding auxillary data
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
