#pragma once

#ifdef ENABLE_HIP

#include "PatchEnergyJITUnion.h"
#include "GPUEvalFactory.h"
#include "hoomd/managed_allocator.h"
#include "EvaluatorUnionGPU.cuh"

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
class PYBIND11_EXPORT PatchEnergyJITUnionGPU : public PatchEnergyJITUnion
    {
    public:
        //! Constructor
        PatchEnergyJITUnionGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<ExecutionConfiguration> exec_conf,
            const std::string& llvm_ir_iso, Scalar r_cut_iso,
            const unsigned int array_size_iso,
            const std::string& llvm_ir_union, Scalar r_cut_union,
            const unsigned int array_size_union,
            const std::string& code,
            const std::string& kernel_name,
            const std::string& include_path,
            const std::string& include_path_source,
            const std::string& cuda_devrt_library_path,
            unsigned int compute_arch)
            : PatchEnergyJITUnion(sysdef, exec_conf, llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir_union, r_cut_union, array_size_union),
              m_gpu_factory(exec_conf, code, kernel_name, include_path, include_path_source, cuda_devrt_library_path, compute_arch),
              m_d_union_params(m_sysdef->getParticleData()->getNTypes(), jit::union_params_t(), managed_allocator<jit::union_params_t>(m_exec_conf->isCUDAEnabled()))
            {
            // allocate data array
            cudaMallocManaged(&m_d_alpha, sizeof(float)*m_alpha_size);
            CHECK_CUDA_ERROR();

            // allocate data array for unions
            cudaMallocManaged(&m_d_alpha_union, sizeof(float)*m_alpha_size_union);
            CHECK_CUDA_ERROR();

            m_gpu_factory.setAlphaPtr(&m_alpha.front());
            m_gpu_factory.setAlphaUnionPtr(&m_alpha.front());
            m_gpu_factory.setUnionParamsPtr(&m_d_union_params.front());
            }

        virtual ~PatchEnergyJITUnionGPU()
            {
            cudaFree(m_d_alpha);
            CHECK_CUDA_ERROR();
            cudaFree(m_d_alpha_union);
            CHECK_CUDA_ERROR();
            }

        //! Set the per-type constituent particles
        /*! \param type The particle type to set the constituent particles for
            \param rcut The maximum cutoff over all constituent particles for this type
            \param types The type IDs for every constituent particle
            \param positions The positions
            \param orientations The orientations
            \param leaf_capacity Number of particles in OBB tree leaf
         */
        virtual void setParam(unsigned int type,
            pybind11::list types,
            pybind11::list positions,
            pybind11::list orientations,
            pybind11::list diameters,
            pybind11::list charges,
            unsigned int leaf_capacity=4);

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
        virtual void launchKernel(unsigned int idev, dim3 grid, dim3 threads, unsigned int sharedMemBytes, hipStream_t hStream, void** kernelParams) const
            {
            m_gpu_factory.launchKernel(idev, grid, threads, sharedMemBytes, hStream, kernelParams);
            }

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            PatchEnergyJITUnion::slotNumTypesChange();
            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_d_union_params.resize(ntypes);

            // update device side pointer
            m_gpu_factory.setUnionParamsPtr(&m_d_union_params.front());
            }

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation
        float *m_d_alpha;                                   //!< device memory holding auxillary data
        float *m_d_alpha_union;                             //!< device memory holding auxillary data
        std::vector<jit::union_params_t, managed_allocator<jit::union_params_t> > m_d_union_params;   //!< Parameters for each particle type on GPU
    };

//! Exports the PatchEnergyJITUnionGPU class to python
void export_PatchEnergyJITUnionGPU(pybind11::module &m);
#endif
