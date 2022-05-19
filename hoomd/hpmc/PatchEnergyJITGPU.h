// Copyright (c) 2009-2022 The Regents of the University of Michigan.
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

        // tuning params for patch narrow phase
        std::vector<unsigned int> valid_params_patch;
        const unsigned int narrow_phase_max_threads_per_eval = this->m_exec_conf->dev_prop.warpSize;
        auto& launch_bounds = m_gpu_factory.getLaunchBounds();
        for (auto cur_launch_bounds : launch_bounds)
            {
            for (unsigned int group_size = 1; group_size <= cur_launch_bounds; group_size *= 2)
                {
                for (unsigned int eval_threads = 1;
                     eval_threads <= narrow_phase_max_threads_per_eval;
                     eval_threads *= 2)
                    {
                    if ((cur_launch_bounds % (group_size * eval_threads)) == 0)
                        valid_params_patch.push_back(cur_launch_bounds * 1000000 + group_size * 100
                                                     + eval_threads);
                    }
                }
            }

        m_tuner_narrow_patch.reset(
            new Autotuner(valid_params_patch, 5, 100000, "hpmc_narrow_patch", this->m_exec_conf));
        }

    //! Asynchronously launch the JIT kernel
    /*! \param args Kernel arguments
        \param hStream stream to execute on
        */
    virtual void computePatchEnergyGPU(const gpu_args_t& args, hipStream_t hStream);

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        m_tuner_narrow_patch->start();
        }

    protected:
    std::unique_ptr<Autotuner> m_tuner_narrow_patch; //!< Autotuner for the narrow phase

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
