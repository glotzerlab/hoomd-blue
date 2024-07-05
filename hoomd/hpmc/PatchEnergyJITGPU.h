// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _PATCH_ENERGY_JIT_GPU_H_
#define _PATCH_ENERGY_JIT_GPU_H_

#ifdef ENABLE_HIP

#include "GPUEvalFactory.h"
#include "PatchEnergyJIT.h"
#include <pybind11/stl.h>

#include <vector>

#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace hpmc
    {
//! Evaluate patch energies via runtime generated code, GPU version
class PYBIND11_EXPORT PatchEnergyJITGPU : public PatchEnergyJIT
    {
    public:
    //! Constructor
    PatchEnergyJITGPU(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<ExecutionConfiguration> exec_conf,
                      const std::string& cpu_code,
                      const std::vector<std::string>& cpu_compiler_args,
                      Scalar r_cut,
                      pybind11::array_t<float> param_array,
                      const std::string& gpu_code,
                      const std::string& kernel_name,
                      const std::vector<std::string>& options,
                      const std::string& cuda_devrt_library_path,
                      unsigned int compute_arch)
        : PatchEnergyJIT(sysdef, exec_conf, cpu_code, cpu_compiler_args, r_cut, param_array),
          m_gpu_factory(exec_conf,
                        gpu_code,
                        kernel_name,
                        options,
                        cuda_devrt_library_path,
                        compute_arch)
        {
        m_gpu_factory.setAlphaPtr(&m_param_array.front(), this->m_is_union);

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

            return (group_size <= block_size) && (block_size % (group_size * eval_threads)) == 0;
        };

        auto& launch_bounds = m_gpu_factory.getLaunchBounds();
        std::vector<unsigned int> valid_group_size = launch_bounds;
        // add subwarp group sizes
        for (unsigned int i = this->m_exec_conf->dev_prop.warpSize / 2; i >= 1; i /= 2)
            valid_group_size.insert(valid_group_size.begin(), i);

        m_tuner_narrow_patch.reset(new Autotuner<3>(
            {launch_bounds, valid_group_size, AutotunerBase::getTppListPow2(m_exec_conf)},
            this->m_exec_conf,
            "hpmc_narrow_patch",
            3,
            false,
            is_parameter_valid));

        m_autotuners.push_back(m_tuner_narrow_patch);
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
    };

namespace detail
    {
//! Exports the PatchEnergyJIT class to python
inline void export_PatchEnergyJITGPU(pybind11::module& m)
    {
    pybind11::class_<PatchEnergyJITGPU, PatchEnergyJIT, std::shared_ptr<PatchEnergyJITGPU>>(
        m,
        "PatchEnergyJITGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExecutionConfiguration>,
                            const std::string&,
                            const std::vector<std::string>&,
                            Scalar,
                            pybind11::array_t<float>,
                            const std::string&,
                            const std::string&,
                            const std::vector<std::string>&,
                            const std::string&,
                            unsigned int>());
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
#endif
#endif // _PATCH_ENERGY_JIT_GPU_H_
