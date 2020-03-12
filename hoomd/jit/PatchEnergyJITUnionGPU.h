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
            const std::vector<std::string>& include_paths,
            const std::string& cuda_devrt_library_path,
            unsigned int compute_arch)
            : PatchEnergyJITUnion(sysdef, exec_conf, llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir_union, r_cut_union, array_size_union),
              m_gpu_factory(exec_conf, code, kernel_name, include_paths, cuda_devrt_library_path, compute_arch),
              m_d_union_params(m_sysdef->getParticleData()->getNTypes(), jit::union_params_t(), managed_allocator<jit::union_params_t>(m_exec_conf->isCUDAEnabled())),
              m_params_updated(false)
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
            m_gpu_factory.setRCutUnion(m_rcut_union);
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
           \param eval_threads template parameter
           \param launch_bounds template parameter
         */
        virtual unsigned int getKernelMaxThreads(unsigned int idev, unsigned int eval_threads, unsigned int launch_bounds)
            {
            return m_gpu_factory.getKernelMaxThreads(idev, eval_threads, launch_bounds);
            }

        //! Return the shared size usage in bytes for this kernel
        /* \param idev the logical GPU id
           \param eval_threads Template parameter
           \param launch_bounds Template parameter
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
            \param extra_bytes Maximum extra bytes of shared memory (modifiable value passed to kernel)
            \param eval_threads Number of threads to use for energy evaluation
            \param launch_bounds Selected launch bounds (template parameter)
            */
        virtual void launchKernel(unsigned int idev, dim3 grid, dim3 threads,
            unsigned int sharedMemBytes, hipStream_t hStream,
            void** kernelParams, unsigned int &max_extra_bytes,
            unsigned int eval_threads, unsigned int launch_bounds)
            {
            // add shape data structures to shared memory requirements
            sharedMemBytes += m_d_union_params.size()*sizeof(jit::union_params_t);

            // allocate some extra shared mem to store union shape parameters
            bool init_shared_bytes = m_base_shared_bytes.find(std::make_pair(eval_threads, launch_bounds)) != m_base_shared_bytes.end();
            unsigned int kernel_shared_bytes = getKernelSharedSize(idev, eval_threads, launch_bounds);
            bool shared_bytes_changed = init_shared_bytes || (m_base_shared_bytes[std::make_pair(eval_threads, launch_bounds)] != sharedMemBytes + kernel_shared_bytes);
            m_base_shared_bytes[std::make_pair(eval_threads,launch_bounds)] = sharedMemBytes + kernel_shared_bytes;

            max_extra_bytes = m_exec_conf->dev_prop.sharedMemPerBlock - m_base_shared_bytes[std::make_pair(eval_threads,launch_bounds)];
            bool init_extra_bytes = m_extra_bytes.find(std::make_pair(eval_threads,launch_bounds)) != m_extra_bytes.end();
            if (init_extra_bytes || m_params_updated || shared_bytes_changed)
                {
                // required for memory coherency
                hipDeviceSynchronize();

                // determine dynamically requested shared memory
                char *ptr = (char *)nullptr;
                unsigned int available_bytes = max_extra_bytes;
                for (unsigned int i = 0; i < m_d_union_params.size(); ++i)
                    {
                    m_d_union_params[i].allocate_shared(ptr, available_bytes);
                    }
                m_extra_bytes[std::make_pair(eval_threads,launch_bounds)] = max_extra_bytes - available_bytes;
                }

            m_params_updated = false;

            sharedMemBytes += m_extra_bytes[std::make_pair(eval_threads,launch_bounds)];

            // launch kernel
            m_gpu_factory.launchKernel(idev, grid, threads, sharedMemBytes, hStream, kernelParams, eval_threads, launch_bounds);
            }

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            PatchEnergyJITUnion::slotNumTypesChange();
            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_d_union_params.resize(ntypes);

            // update device side pointer
            m_gpu_factory.setUnionParamsPtr(&m_d_union_params.front());
            m_params_updated = true;
            }

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation
        float *m_d_alpha;                                   //!< device memory holding auxillary data
        float *m_d_alpha_union;                             //!< device memory holding auxillary data
        std::vector<jit::union_params_t, managed_allocator<jit::union_params_t> > m_d_union_params;   //!< Parameters for each particle type on GPU
        bool m_params_updated;                              //!< True if parameters have been updated

        std::map<std::pair<unsigned int, unsigned int>, unsigned int> m_base_shared_bytes;      //!< Kernel shared memory, for every template
        std::map<std::pair<unsigned int, unsigned int>, unsigned int> m_extra_bytes;            //!< Kernel extra shared bytes, for every template
    };

//! Exports the PatchEnergyJITUnionGPU class to python
void export_PatchEnergyJITUnionGPU(pybind11::module &m);
#endif
