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
                       const std::string& kernel_name,
                       const std::string& include_path,
                       const std::string& include_path_source,
                       const std::string& cuda_devrt_library_path,
                       unsigned int compute_arch)
            : PatchEnergyJIT(exec_conf, llvm_ir, r_cut, array_size),
              m_gpu_factory(exec_conf, code, kernel_name, include_path, include_path_source, cuda_devrt_library_path, compute_arch)
            {
            m_gpu_factory.setAlphaPtr(&m_alpha.front());
            }

        //! Return the __global__ function pointer for a GPU
        /* \param idev the logical GPU id
         */
        virtual const void *getKernelAddress(unsigned int idev) const
            {
            return m_gpu_factory.getKernelAddress(idev);
            }

        //! Return the maximum number of threads per block for this kernel
        /* \param idev the logical GPU id
         */
        virtual unsigned int getKernelMaxThreads(unsigned int idev) const
            {
            return m_gpu_factory.getKernelMaxThreads(idev);
            }

        //! Return the shared size usage in bytes for this kernel
        /* \param idev the logical GPU id
         */
        virtual unsigned int getKernelSharedSize(unsigned int idev) const
            {
            return m_gpu_factory.getKernelSharedSize(idev);
            }

        //! Asynchronously launch the JIT kernel
        /*! \param idev logical GPU id to launch on
            \param grid The grid dimensions
            \param threads The thread block dimensions
            \param sharedMemBytes The size of the dynamic shared mem allocation
            \param hStream stream to execute on
            \param kernelParams the kernel parameters
            */
        virtual void launchKernel(unsigned int idev, dim3 grid, dim3 threads, unsigned int sharedMemBytes, hipStream_t hStream, void** kernelParams)
            {
            m_gpu_factory.launchKernel(idev, grid, threads, sharedMemBytes, hStream, kernelParams);
            }

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation
    };

//! Exports the PatchEnergyJIT class to python
inline void export_PatchEnergyJITGPU(pybind11::module &m)
    {
    pybind11::class_<PatchEnergyJITGPU, PatchEnergyJIT, std::shared_ptr<PatchEnergyJITGPU> >(m, "PatchEnergyJITGPU")
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar, const unsigned int, const std::string&, const std::string&,
                                 const std::string&, const std::string&, const std::string&,
                                 unsigned int >())
            ;
    }
#endif
#endif // _PATCH_ENERGY_JIT_GPU_H_
