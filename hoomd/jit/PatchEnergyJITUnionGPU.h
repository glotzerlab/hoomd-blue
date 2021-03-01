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
            m_gpu_factory.setRCutUnion(float(m_rcut_union));

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

            m_managed_memory = true;
            }

        virtual ~PatchEnergyJITUnionGPU() {}

        //! Set per-type typeid of constituent particles
        virtual void setTypeids(std::string type, pybind11::list typeids)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            unsigned int N = (unsigned int) pybind11::len(typeids);
            m_type[pid].resize(N);

            jit::union_params_t params(N, true);
            if(m_d_union_params[pid].N == N)
                {
                params = m_d_union_params[pid];
                }

            for (unsigned int i = 0; i < N; i++)
                {
                unsigned int t = pybind11::cast<unsigned int>(typeids[i]);
                m_type[pid][i] = t;
                params.mtype[i] = t;
                }
            // store result
            m_d_union_params[pid] = params;
            // cudaMemadviseReadMostly
            m_d_union_params[pid].set_memory_hint();

            }

        //! Set per-type positions of the constituent particles
        virtual void setPositions(std::string type, pybind11::list position)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            unsigned int N = (unsigned int) pybind11::len(position);
            m_position[pid].resize(N);

            jit::union_params_t params(N, true);
            if(m_d_union_params[pid].N == N)
                {
                params = m_d_union_params[pid];
                }

            for (unsigned int i = 0; i < N; i++)
                {
                pybind11::tuple p_i = position[i];
                vec3<float> pos(p_i[0].cast<float>(),
                                p_i[1].cast<float>(),
                                p_i[2].cast<float>());
                m_position[pid][i] = pos;
                params.mpos[i] = pos;
                }

            if (std::find(m_updated_types.begin(), m_updated_types.end(), pid) == m_updated_types.end())
                {
                m_updated_types.push_back(pid);
                }

            m_build_obb = true;
            // store result
            m_d_union_params[pid] = params;
            // cudaMemadviseReadMostly
            m_d_union_params[pid].set_memory_hint();

            }

        //! Set per-type positions of the constituent particles
        virtual void setOrientations(std::string type, pybind11::list orientation)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            unsigned int N = (unsigned int) pybind11::len(orientation);
            m_orientation[pid].resize(N);

            jit::union_params_t params(N, true);
            if(m_d_union_params[pid].N == N)
                {
                params = m_d_union_params[pid];
                }

            for (unsigned int i = 0; i < N; i++)
                {
                pybind11::tuple q_i = orientation[i];
                float s = q_i[0].cast<float>();
                float x = q_i[1].cast<float>();
                float y = q_i[2].cast<float>();
                float z = q_i[3].cast<float>();
                quat<float> ort(s, vec3<float>(x,y,z));
                m_orientation[pid][i] = ort;
                params.morientation[i] = ort;
                }

            // store result
            m_d_union_params[pid] = params;
            // cudaMemadviseReadMostly
            m_d_union_params[pid].set_memory_hint();

            }

        //! Set per-type diameters of the constituent particles
        virtual void setDiameters(std::string type, pybind11::list diameter)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            unsigned int N = (unsigned int) pybind11::len(diameter);
            m_diameter[pid].resize(N);

            jit::union_params_t params(N, true);
            if(m_d_union_params[pid].N == N)
                {
                params = m_d_union_params[pid];
                }

            for (unsigned int i = 0; i < N; i++)
                {
                float d = diameter[i].cast<float>();
                m_diameter[pid][i] = d;
                params.mdiameter[i] = d;
                }

            if (std::find(m_updated_types.begin(), m_updated_types.end(), pid) == m_updated_types.end())
                {
                m_updated_types.push_back(pid);
                }

            m_build_obb = true;
            // store result
            m_d_union_params[pid] = params;
            // cudaMemadviseReadMostly
            m_d_union_params[pid].set_memory_hint();

            }

        //! Set per-type charges of the constituent particles
        virtual void setCharges(std::string type, pybind11::list charge)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            unsigned int N = (unsigned int) pybind11::len(charge);
            m_charge[pid].resize(N);

            jit::union_params_t params(N, true);
            if(m_d_union_params[pid].N == N)
                {
                params = m_d_union_params[pid];
                }

            for (unsigned int i = 0; i < N; i++)
                {
                float q = charge[i].cast<float>();
                m_charge[pid][i] = q;
                params.mcharge[i] = q;
                }

            // store result
            m_d_union_params[pid] = params;
            // cudaMemadviseReadMostly
            m_d_union_params[pid].set_memory_hint();

            }

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
