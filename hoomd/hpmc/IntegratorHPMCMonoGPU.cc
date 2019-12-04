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
void hpmc_narrow_phase_patch(const hpmc_args_t& args, const hpmc_patch_args_t& patch_args, const PatchEnergy& patch)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = std::min(args.block_size, (unsigned int)patch.getKernelMaxThreads(0)); // fixme GPU 0

    unsigned int tpp = std::min(args.tpp,run_block_size);
    unsigned int n_groups = run_block_size/tpp;
    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar);

    unsigned int shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar))
        + max_queue_size * 2 * sizeof(unsigned int)
        + min_shared_bytes;

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    unsigned int kernel_shared_bytes = patch.getKernelSharedSize(0); //fixme GPU 0
    while (shared_bytes + kernel_shared_bytes >= args.devprop.sharedMemPerBlock)
        {
        run_block_size -= args.devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = std::min(tpp, run_block_size);
        n_groups = run_block_size / tpp;
        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;
        }

    unsigned int max_extra_bytes = 0;
    #if 0
    // determine dynamically allocated shared memory size
    static unsigned int base_shared_bytes = UINT_MAX;
    bool shared_bytes_changed = base_shared_bytes != shared_bytes + attr.sharedSizeBytes;
    base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
    static unsigned int extra_bytes = UINT_MAX;
    if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
        {
        // required for memory coherency
        cudaDeviceSynchronize();

        // determine dynamically requested shared memory
        char *ptr = (char *)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        extra_bytes = max_extra_bytes - available_bytes;
        }

    shared_bytes += extra_bytes;
    #endif

    dim3 thread(tpp, n_groups, 1);

    for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = args.gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        unsigned int N_old = args.N + args.N_ghost;
        void *k_args[] = {(void *)&args.d_postype, (void *)&args.d_orientation, (void *)&args.d_trial_postype, (void *) &args.d_trial_orientation,
            (void *) &patch_args.d_charge, (void *) &patch_args.d_diameter, (void *) &args.d_excell_idx, (void *) &args.d_excell_size, (void *) &args.excli,
            (void *) &patch_args.d_nlist, (void *) &patch_args.d_energy, (void *) &patch_args.d_nneigh, (void *) &patch_args.maxn,  (void *) &args.num_types,
            (void *) &args.box, (void *) &args.ghost_width, (void *) &args.cell_dim, (void *) &args.ci, (void *) &N_old, (void *) &args.N,
            (void *) &patch_args.old_config, (void *) &patch_args.r_cut_patch, (void *) &patch_args.d_additive_cutoff,
            (void *) &patch_args.d_overflow, (void *) &max_extra_bytes, (void *) &max_queue_size, (void *)&range.first, (void *) &nwork};

        patch.launchKernel(idev, grid, thread, shared_bytes, 0, k_args);
        }
    }

} // end namespace gpu
} // end namespace hpmc

#endif
