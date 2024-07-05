// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP
#include "PatchEnergyJITUnionGPU.h"

#include "hoomd/hpmc/EvaluatorUnionGPU.cuh"
#include <pybind11/stl.h>

namespace hoomd
    {
namespace hpmc
    {
//! Kernel driver for kernel::hpmc_narrow_phase_patch
void PatchEnergyJITUnionGPU::computePatchEnergyGPU(const gpu_args_t& args, hipStream_t hStream)
    {
#ifdef __HIP_PLATFORM_NVCC__
    assert(args.d_postype);
    assert(args.d_orientation);

    auto param = m_tuner_narrow_patch->getParam();
    unsigned int block_size = param[0];
    unsigned int req_tpp = param[1];
    unsigned int eval_threads = param[2];

    this->m_exec_conf->beginMultiGPU();
    m_tuner_narrow_patch->begin();

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic
    // shared memory usage
    unsigned int run_block_size
        = std::min(block_size,
                   m_gpu_factory.getKernelMaxThreads(0, eval_threads, block_size)); // fixme GPU 0

    unsigned int tpp = std::min(req_tpp, run_block_size);
    while (eval_threads * tpp > run_block_size || run_block_size % (eval_threads * tpp) != 0)
        {
        tpp--;
        }
    auto& devprop = m_exec_conf->dev_prop;
    tpp = std::min((unsigned int)devprop.maxThreadsDim[2], tpp);

    unsigned int n_groups = run_block_size / (tpp * eval_threads);
    unsigned int max_queue_size = n_groups * tpp;

    size_t min_shared_bytes
        = args.num_types * sizeof(Scalar) + m_d_union_params.size() * sizeof(jit::union_params_t);

    size_t shared_bytes = n_groups
                              * (sizeof(unsigned int) + 2 * sizeof(Scalar4) + 2 * sizeof(Scalar3)
                                 + 2 * sizeof(Scalar) + 2 * sizeof(float))
                          + max_queue_size * 2 * sizeof(unsigned int) + min_shared_bytes;

    if (min_shared_bytes >= devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of "
                                 "particle types or size of shape parameters");

    size_t kernel_shared_bytes
        = m_gpu_factory.getKernelSharedSize(0, eval_threads, block_size); // fixme GPU 0
    while (shared_bytes + kernel_shared_bytes >= devprop.sharedMemPerBlock)
        {
        run_block_size -= devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = std::min(req_tpp, run_block_size);
        while (eval_threads * tpp > run_block_size || run_block_size % (eval_threads * tpp) != 0)
            {
            tpp--;
            }
        tpp = std::min((unsigned int)devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

        n_groups = run_block_size / (tpp * eval_threads);

        max_queue_size = n_groups * tpp;

        shared_bytes = n_groups
                           * (sizeof(unsigned int) + 2 * sizeof(Scalar4) + 2 * sizeof(Scalar3)
                              + 2 * sizeof(Scalar) + 2 * sizeof(float))
                       + max_queue_size * 2 * sizeof(unsigned int) + min_shared_bytes;
        }

    // allocate some extra shared mem to store union shape parameters
    unsigned int max_extra_bytes = (unsigned int)(m_exec_conf->dev_prop.sharedMemPerBlock
                                                  - shared_bytes - kernel_shared_bytes);

    // determine dynamically requested shared memory
    char* ptr = (char*)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < m_d_union_params.size(); ++i)
        {
        m_d_union_params[i].allocate_shared(ptr, available_bytes);
        }
    unsigned int extra_bytes = max_extra_bytes - available_bytes;
    shared_bytes += extra_bytes;

    dim3 thread(eval_threads, n_groups, tpp);

    auto& gpu_partition = args.gpu_partition;

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1) / n_groups;

        dim3 grid(num_blocks, 1, 1);

        // configure the kernel
        auto launcher = m_gpu_factory.configureKernel(idev,
                                                      grid,
                                                      thread,
                                                      shared_bytes,
                                                      hStream,
                                                      eval_threads,
                                                      block_size);

        CUresult res = launcher(args.d_postype,
                                args.d_orientation,
                                args.d_trial_postype,
                                args.d_trial_orientation,
                                args.d_trial_move_type,
                                args.d_charge,
                                args.d_diameter,
                                args.d_excell_idx,
                                args.d_excell_size,
                                args.excli,
                                args.d_update_order_by_ptl,
                                args.d_reject_in,
                                args.d_reject_out,
                                args.seed,
                                args.timestep,
                                args.select,
                                args.rank,
                                args.num_types,
                                args.box,
                                args.ghost_width,
                                args.cell_dim,
                                args.ci,
                                args.N,
                                args.r_cut_patch,
                                args.d_additive_cutoff,
                                args.d_reject_out_of_cell,
                                max_queue_size,
                                range.first,
                                nwork,
                                max_extra_bytes);

        if (res != CUDA_SUCCESS)
            {
            char* error;
            cuGetErrorString(res, const_cast<const char**>(&error));
            throw std::runtime_error("Error launching NVRTC kernel: " + std::string(error));
            }
        }

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_narrow_patch->end();
    m_exec_conf->endMultiGPU();
#endif
    }

namespace detail
    {
void export_PatchEnergyJITUnionGPU(pybind11::module& m)
    {
    pybind11::class_<PatchEnergyJITUnionGPU,
                     PatchEnergyJITUnion,
                     std::shared_ptr<PatchEnergyJITUnionGPU>>(m, "PatchEnergyJITUnionGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExecutionConfiguration>,
                            const std::string&,
                            const std::vector<std::string>&,
                            Scalar,
                            pybind11::array_t<float>,
                            const std::string&,
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
