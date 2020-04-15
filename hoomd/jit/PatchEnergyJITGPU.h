#ifndef _PATCH_ENERGY_JIT_GPU_H_
#define _PATCH_ENERGY_JIT_GPU_H_

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"
#include "GPUEvalFactory.h"
#include <pybind11/stl.h>

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
                       const std::vector<std::string>& include_paths,
                       const std::string& cuda_devrt_library_path,
                       unsigned int compute_arch)
            : PatchEnergyJIT(exec_conf, llvm_ir, r_cut, array_size),
              m_gpu_factory(exec_conf, code, kernel_name, include_paths, cuda_devrt_library_path, compute_arch)
            {
            m_gpu_factory.setAlphaPtr(&m_alpha.front());
            }

        //! Return the list of available launch bounds
        /* \param idev the logical GPU id
           \param eval_threads template parameter
         */
        virtual const std::vector<unsigned int>& getLaunchBounds() const
            {
            return m_gpu_factory.getLaunchBounds();
            }

        //! Return the maximum number of threads per block for this kernel
        /* \param idev the logical GPU id
           \param eval_threads kernel template parameter
           \param launch_bounds launch bounds, template parameter
         */
        virtual unsigned int getKernelMaxThreads(unsigned int idev, unsigned int eval_threads, unsigned int launch_bounds)
            {
            return m_gpu_factory.getKernelMaxThreads(idev, eval_threads, launch_bounds);
            }

        //! Return the shared size usage in bytes for this kernel
        /* \param idev the logical GPU id
           \param eval_threads template parameter
           \param launch_bounds template parameter
         */
        virtual unsigned int getKernelSharedSize(unsigned int idev, unsigned int eval_threads, unsigned int launch_bounds)
            {
            return m_gpu_factory.getKernelSharedSize(idev, eval_threads, launch_bounds);
            }

        //! Asynchronously launch the JIT kernel
        /*! \param idev logical GPU id to launch on
            \param grid The grid dimensions
            \param threads The thread block dimensions
            \param sharedMemBytes The size of the dynamic shared mem allocation
            \param hStream stream to execute on
            \param kernelParams the kernel parameters
            \param max_extra_bytes Maximum extra bytes of shared memory, kernel argument
            \param eval_threads template parameter
            \param launch_bouds template parameter
            */
        virtual void launchKernel(unsigned int idev, dim3 grid, dim3 threads,
            unsigned int sharedMemBytes, hipStream_t hStream,
            void** kernelParams, unsigned int& max_extra_bytes,
            unsigned int eval_threads, unsigned int launch_bounds)
            {
            m_gpu_factory.launchKernel(idev, grid, threads, sharedMemBytes, hStream, kernelParams, eval_threads, launch_bounds);
            }

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation
    };

//! Exports the PatchEnergyJIT class to python
inline void export_PatchEnergyJITGPU(pybind11::module &m)
    {
    pybind11::class_<PatchEnergyJITGPU, PatchEnergyJIT, std::shared_ptr<PatchEnergyJITGPU> >(m, "PatchEnergyJITGPU")
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&,
                                 const std::string&,
                                 const std::vector<std::string>&,
                                 const std::string&,
                                 unsigned int >())
            ;
    }
#endif
#endif // _PATCH_ENERGY_JIT_GPU_H_
