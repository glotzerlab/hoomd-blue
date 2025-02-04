// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP

#include "EvaluatorUnionGPU.cuh"
#include "GPUEvalFactory.h"
#include "PatchEnergyJITUnion.h"
#include "hoomd/managed_allocator.h"

#include <vector>

namespace hoomd
    {
namespace hpmc
    {
//! Evaluate patch energies via runtime generated code, GPU version
class PYBIND11_EXPORT PatchEnergyJITUnionGPU : public PatchEnergyJITUnion
    {
    public:
    //! Constructor
    PatchEnergyJITUnionGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ExecutionConfiguration> exec_conf,
                           const std::string& cpu_code_iso,
                           const std::vector<std::string>& cpu_compiler_args,
                           Scalar r_cut_iso,
                           pybind11::array_t<float> param_array_isotropic,
                           const std::string& cpu_code_constituent,
                           Scalar r_cut_constituent,
                           pybind11::array_t<float> param_array_constituent,
                           const std::string& code,
                           const std::string& kernel_name,
                           const std::vector<std::string>& options,
                           const std::string& cuda_devrt_library_path,
                           unsigned int compute_arch)
        : PatchEnergyJITUnion(sysdef,
                              exec_conf,
                              cpu_code_iso,
                              cpu_compiler_args,
                              r_cut_iso,
                              param_array_isotropic,
                              cpu_code_constituent,
                              r_cut_constituent,
                              param_array_constituent),
          m_gpu_factory(exec_conf,
                        code,
                        kernel_name,
                        options,
                        cuda_devrt_library_path,
                        compute_arch),
          m_d_union_params(
              m_sysdef->getParticleData()->getNTypes(),
              jit::union_params_t(),
              hoomd::detail::managed_allocator<jit::union_params_t>(m_exec_conf->isCUDAEnabled()))
        {
        m_gpu_factory.setAlphaPtr(m_param_array.data(), this->m_is_union);
        m_gpu_factory.setAlphaUnionPtr(m_param_array_constituent.data());
        m_gpu_factory.setUnionParamsPtr(m_d_union_params.data());
        m_gpu_factory.setRCutUnion(float(m_r_cut_constituent));

        // Autotuner parameters:
        // 0: block size
        // 1: group size
        // 2: eval threads

        std::function<bool(const std::array<unsigned int, 3>&)> is_parameter_valid
            = [](const std::array<unsigned int, 3>& parameter) -> bool
        {
            unsigned int block_size = parameter[0];
            unsigned int group_size = parameter[1];
            unsigned int eval_threads = parameter[2];

            return (group_size <= block_size) && (block_size % (eval_threads * group_size)) == 0;
        };

        auto& launch_bounds = m_gpu_factory.getLaunchBounds();
        std::vector<unsigned int> valid_group_size = launch_bounds;
        // add subwarp group sizes
        for (unsigned int i = this->m_exec_conf->dev_prop.warpSize / 2; i >= 1; i /= 2)
            valid_group_size.insert(valid_group_size.begin(), i);

        m_tuner_narrow_patch.reset(new Autotuner<3>(
            {launch_bounds, valid_group_size, AutotunerBase::getTppListPow2(m_exec_conf)},
            this->m_exec_conf,
            "hpmc_narrow_patch_jit",
            3,
            false,
            is_parameter_valid));

        m_autotuners.push_back(m_tuner_narrow_patch);

        m_managed_memory = true;
        }

    virtual ~PatchEnergyJITUnionGPU() { }

    virtual void buildOBBTree(unsigned int type_id)
        {
        PatchEnergyJITUnion::buildOBBTree(type_id);
        m_d_union_params[type_id].tree = m_tree[type_id];
        }

    //! Set per-type typeid of constituent particles
    virtual void setTypeids(std::string type, pybind11::list typeids)
        {
        PatchEnergyJITUnion::setTypeids(type, typeids);
        unsigned int type_id = m_sysdef->getParticleData()->getTypeByName(type);
        auto N = static_cast<unsigned int>(m_type[type_id].size());
        ManagedArray<unsigned int> new_type_ids(N, true);
        std::copy(m_type[type_id].begin(), m_type[type_id].end(), new_type_ids.get());

        // store result
        m_d_union_params[type_id].mtype = new_type_ids;
        // cudaMemadviseReadMostly
        m_d_union_params[type_id].set_memory_hint();
        }

    //! Set per-type positions of the constituent particles
    virtual void setPositions(std::string type, pybind11::list position)
        {
        PatchEnergyJITUnion::setPositions(type, position);
        unsigned int type_id = m_sysdef->getParticleData()->getTypeByName(type);
        auto N = static_cast<unsigned int>(m_position[type_id].size());
        ManagedArray<vec3<float>> new_positions(N, true);
        std::copy(m_position[type_id].begin(), m_position[type_id].end(), new_positions.get());

        // store result
        m_d_union_params[type_id].mpos = new_positions;
        // cudaMemadviseReadMostly
        m_d_union_params[type_id].set_memory_hint();
        }

    //! Set per-type positions of the constituent particles
    virtual void setOrientations(std::string type, pybind11::list orientation)
        {
        PatchEnergyJITUnion::setOrientations(type, orientation);
        unsigned int type_id = m_sysdef->getParticleData()->getTypeByName(type);
        auto N = static_cast<unsigned int>(m_orientation[type_id].size());
        ManagedArray<quat<float>> new_orientations(N, true);
        std::copy(m_orientation[type_id].begin(),
                  m_orientation[type_id].end(),
                  new_orientations.get());

        // store result
        m_d_union_params[type_id].morientation = new_orientations;
        // cudaMemadviseReadMostly
        m_d_union_params[type_id].set_memory_hint();
        }

    //! Set per-type diameters of the constituent particles
    virtual void setDiameters(std::string type, pybind11::list diameter)
        {
        PatchEnergyJITUnion::setDiameters(type, diameter);
        unsigned int type_id = m_sysdef->getParticleData()->getTypeByName(type);
        auto N = static_cast<unsigned int>(m_diameter[type_id].size());
        ManagedArray<float> new_diameters(N, true);
        std::copy(m_diameter[type_id].begin(), m_diameter[type_id].end(), new_diameters.get());

        // store result
        m_d_union_params[type_id].mdiameter = new_diameters;
        // cudaMemadviseReadMostly
        m_d_union_params[type_id].set_memory_hint();
        }

    //! Set per-type charges of the constituent particles
    virtual void setCharges(std::string type, pybind11::list charge)
        {
        PatchEnergyJITUnion::setCharges(type, charge);
        unsigned int type_id = m_sysdef->getParticleData()->getTypeByName(type);
        auto N = static_cast<unsigned int>(m_charge[type_id].size());
        ManagedArray<float> new_charges(N, true);
        std::copy(m_charge[type_id].begin(), m_charge[type_id].end(), new_charges.get());

        // store result
        m_d_union_params[type_id].mcharge = new_charges;
        // cudaMemadviseReadMostly
        m_d_union_params[type_id].set_memory_hint();
        }

    //! Asynchronously launch the JIT kernel
    /*! \param args Kernel arguments
        \param hStream stream to execute on
        */
    virtual void computePatchEnergyGPU(const gpu_args_t& args, hipStream_t hStream);

    protected:
    /// Autotuner for the narrow phase kernel.
    std::shared_ptr<Autotuner<3>> m_tuner_narrow_patch;

    private:
    GPUEvalFactory m_gpu_factory; //!< JIT implementation

    std::vector<jit::union_params_t, hoomd::detail::managed_allocator<jit::union_params_t>>
        m_d_union_params; //!< Parameters for each particle type on GPU
    };

namespace detail
    {
//! Exports the PatchEnergyJITUnionGPU class to python
void export_PatchEnergyJITUnionGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
#endif
