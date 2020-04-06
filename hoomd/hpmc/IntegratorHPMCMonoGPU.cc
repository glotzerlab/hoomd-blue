// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "IntegratorHPMC.h"
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoGPU.cuh"
#include <algorithm>

namespace hpmc
{
namespace gpu
{

//! Kernel driver for kernel::hpmc_narrow_phase_patch
void hpmc_narrow_phase_patch(const hpmc_args_t& args, const hpmc_patch_args_t& patch_args, PatchEnergy& patch)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int eval_threads = patch_args.eval_threads;
    unsigned int run_block_size = std::min(args.block_size, (unsigned int)patch.getKernelMaxThreads(0, eval_threads, patch_args.launch_bounds)); // fixme GPU 0

    unsigned int tpp = std::min(args.tpp,run_block_size);
    while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
        {
        tpp--;
        }

    unsigned int n_groups = run_block_size/(tpp*eval_threads);

    // truncate blockDim.z
    n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);
    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar);

    unsigned int shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
        + max_queue_size * 2 * sizeof(unsigned int)
        + min_shared_bytes;

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    unsigned int kernel_shared_bytes = patch.getKernelSharedSize(0, eval_threads, patch_args.launch_bounds); //fixme GPU 0
    while (shared_bytes + kernel_shared_bytes >= args.devprop.sharedMemPerBlock)
        {
        run_block_size -= args.devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = std::min(args.tpp, run_block_size);
        while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
            {
            tpp--;
            }

        n_groups = run_block_size / (tpp*eval_threads);

        // truncate blockDim.z
        n_groups = std::min((unsigned int)args.devprop.maxThreadsDim[2], n_groups);

        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;
        }

    dim3 thread(eval_threads, tpp, n_groups);

    for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = args.gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        unsigned int max_extra_bytes = 0;
        unsigned int N_old = args.N + args.N_ghost;
        void *k_args[] = {(void *)&args.d_postype, (void *)&args.d_orientation, (void *)&args.d_trial_postype, (void *) &args.d_trial_orientation,
            (void *) &patch_args.d_charge, (void *) &patch_args.d_diameter, (void *) &args.d_excell_idx, (void *) &args.d_excell_size, (void *) &args.excli,
            (void *) &patch_args.d_nlist_old, (void *) &patch_args.d_energy_old, (void *) &patch_args.d_nneigh_old,
            (void *) &patch_args.d_nlist_new, (void *) &patch_args.d_energy_new, (void *) &patch_args.d_nneigh_new,
            (void *) &patch_args.maxn,  (void *) &args.num_types,
            (void *) &args.box, (void *) &args.ghost_width, (void *) &args.cell_dim, (void *) &args.ci, (void *) &N_old, (void *) &args.N,
            (void *) &patch_args.r_cut_patch, (void *) &patch_args.d_additive_cutoff,
            (void *) &patch_args.d_overflow, (void *) &max_queue_size, (void *)&range.first, (void *) &nwork, (void *) &max_extra_bytes};

        // launch kernel template
        patch.launchKernel(idev, grid, thread, shared_bytes, 0, k_args, max_extra_bytes, eval_threads, patch_args.launch_bounds);
        }
    }

} // end namespace gpu
} // end namespace hpmc

#endif
