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
            const std::vector<std::string>& options,
            const std::string& cuda_devrt_library_path,
            unsigned int compute_arch)
            : PatchEnergyJITUnion(sysdef, exec_conf, llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir_union, r_cut_union, array_size_union),
              m_gpu_factory(exec_conf, code, kernel_name, options, cuda_devrt_library_path, compute_arch),
              m_d_union_params(m_sysdef->getParticleData()->getNTypes(), jit::union_params_t(), managed_allocator<jit::union_params_t>(m_exec_conf->isCUDAEnabled()))
            {
            m_gpu_factory.setAlphaPtr(&m_alpha.front());
            m_gpu_factory.setAlphaUnionPtr(&m_alpha_union.front());
            m_gpu_factory.setUnionParamsPtr(&m_d_union_params.front());
            m_gpu_factory.setRCutUnion(m_rcut_union);

            // tuning params for patch narrow phase
            std::vector<unsigned int> valid_params_patch;
            const unsigned int narrow_phase_max_threads_per_eval = this->m_exec_conf->dev_prop.warpSize;
            auto& launch_bounds = m_gpu_factory.getLaunchBounds();
            for (auto cur_launch_bounds: launch_bounds)
                {
                for (unsigned int group_size=1; group_size <= cur_launch_bounds; group_size*=2)
                    {
                    for (unsigned int eval_threads=1; eval_threads <= narrow_phase_max_threads_per_eval; eval_threads *= 2)
                        {
                        if ((cur_launch_bounds % (group_size*eval_threads)) == 0)
                            valid_params_patch.push_back(cur_launch_bounds*1000000 + group_size*100 + eval_threads);
                        }
                    }
                }

            m_tuner_narrow_patch.reset(new Autotuner(valid_params_patch, 5, 100000, "hpmc_narrow_patch", this->m_exec_conf));
            }

        virtual ~PatchEnergyJITUnionGPU() {}

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

        //! Asynchronously launch the JIT kernel
        /*! \param args Kernel arguments
            \param hStream stream to execute on
            */
        virtual void computePatchEnergyGPU(const gpu_args_t& args, hipStream_t hStream);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            PatchEnergyJITUnion::slotNumTypesChange();
            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_d_union_params.resize(ntypes);

            // update device side pointer
            m_gpu_factory.setUnionParamsPtr(&m_d_union_params.front());
            }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner_narrow_patch->setPeriod(period);
            m_tuner_narrow_patch->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_narrow_patch;     //!< Autotuner for the narrow phase

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation

        std::vector<jit::union_params_t, managed_allocator<jit::union_params_t> > m_d_union_params;   //!< Parameters for each particle type on GPU
    };

//! Exports the PatchEnergyJITUnionGPU class to python
void export_PatchEnergyJITUnionGPU(pybind11::module &m);
#endif
