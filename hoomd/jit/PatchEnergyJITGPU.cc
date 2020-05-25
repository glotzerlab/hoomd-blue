// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "PatchEnergyJITGPU.h"

//! Kernel driver for kernel::hpmc_narrow_phase_patch
void PatchEnergyJITGPU::computePatchEnergyGPU(const gpu_args_t& args, hipStream_t hStream)
    {
    #ifdef __HIP_PLATFORM_NVCC__
    assert(args.d_postype);
    assert(args.d_orientation);

    unsigned int param = m_tuner_narrow_patch->getParam();
    unsigned int block_size = param/1000000;
    unsigned int req_tpp = (param%1000000)/100;
    unsigned int eval_threads = param % 100;

    this->m_exec_conf->beginMultiGPU();
    m_tuner_narrow_patch->begin();

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = std::min(block_size, m_gpu_factory.getKernelMaxThreads(0, eval_threads, block_size)); // fixme GPU 0

    unsigned int tpp = std::min(req_tpp,run_block_size);
    while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
        {
        tpp--;
        }

    unsigned int n_groups = run_block_size/(tpp*eval_threads);

    // truncate blockDim.z
    auto& devprop = m_exec_conf->dev_prop;
    n_groups = std::min((unsigned int) devprop.maxThreadsDim[2], n_groups);
    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar);

    unsigned int shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
        + max_queue_size * 2 * sizeof(unsigned int)
        + min_shared_bytes;

    if (min_shared_bytes >= devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    unsigned int kernel_shared_bytes = m_gpu_factory.getKernelSharedSize(0, eval_threads, block_size); //fixme GPU 0
    while (shared_bytes + kernel_shared_bytes >= devprop.sharedMemPerBlock)
        {
        run_block_size -= devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = std::min(req_tpp, run_block_size);
        while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
            {
            tpp--;
            }

        n_groups = run_block_size / (tpp*eval_threads);

        // truncate blockDim.z
        n_groups = std::min((unsigned int)devprop.maxThreadsDim[2], n_groups);

        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;
        }

    dim3 thread(eval_threads, tpp, n_groups);

    auto& gpu_partition = args.gpu_partition;

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        unsigned int max_extra_bytes = 0;
        unsigned int N_old = args.N + args.N_ghost;

        // configure the kernel
        auto launcher = m_gpu_factory.configureKernel(idev, grid, thread, shared_bytes, hStream, eval_threads, block_size);

        CUresult res = launcher(args.d_postype,
            args.d_orientation,
            args.d_trial_postype,
            args.d_trial_orientation,
            args.d_charge,
            args.d_diameter,
            args.d_excell_idx,
            args.d_excell_size,
            args.excli,
            args.d_nlist_old,
            args.d_energy_old,
            args.d_nneigh_old,
            args.d_nlist_new,
            args.d_energy_new,
            args.d_nneigh_new,
            args.maxn,
            args.num_types,
            args.box,
            args.ghost_width,
            args.cell_dim,
            args.ci,
            N_old,
            args.N,
            args.r_cut_patch,
            args.d_additive_cutoff,
            args.d_overflow,
            max_queue_size,
            range.first,
            nwork,
            max_extra_bytes);

        if (res != CUDA_SUCCESS)
            {
            char *error;
            cuGetErrorString(res, const_cast<const char **>(&error));
            throw std::runtime_error("Error launching NVRTC kernel: "+std::string(error));
            }
        }

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_narrow_patch->end();
    m_exec_conf->endMultiGPU();
    #endif
    }

#endif
