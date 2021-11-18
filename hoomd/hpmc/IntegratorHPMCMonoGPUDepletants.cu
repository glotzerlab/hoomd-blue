// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPUDepletants.cuh"
#include "hoomd/CachedAllocator.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
namespace kernel
    {
//! Generate number of depletants per particle
__global__ void generate_num_depletants(const uint16_t seed,
                                        const uint64_t timestep,
                                        const unsigned int select,
                                        const unsigned int rank,
                                        const unsigned int depletant_type_a,
                                        const unsigned int depletant_type_b,
                                        const Index2D depletant_idx,
                                        const unsigned int work_offset,
                                        const unsigned int nwork,
                                        const Scalar* d_lambda,
                                        const Scalar4* d_postype,
                                        unsigned int* d_n_depletants)
    {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= nwork)
        return;

    idx += work_offset;

    hoomd::RandomGenerator rng_poisson(
        hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantNum, timestep, seed),
        hoomd::Counter(idx,
                       rank,
                       depletant_idx(depletant_type_a, depletant_type_b),
                       static_cast<uint16_t>(select)));
    unsigned int type_i = __scalar_as_int(d_postype[idx].w);
    d_n_depletants[idx] = hoomd::PoissonDistribution<Scalar>(
        d_lambda[type_i * depletant_idx.getNumElements()
                 + depletant_idx(depletant_type_a, depletant_type_b)])(rng_poisson);
    }

//! Generate number of depletants per particle (ntrial version)
__global__ void generate_num_depletants_ntrial(const Scalar4* d_vel,
                                               const Scalar4* d_trial_vel,
                                               const unsigned int ntrial,
                                               const unsigned int depletant_type_a,
                                               const unsigned int depletant_type_b,
                                               const Index2D depletant_idx,
                                               const Scalar* d_lambda,
                                               const Scalar4* d_postype,
                                               unsigned int* d_n_depletants,
                                               const unsigned int N_local,
                                               const unsigned int work_offset,
                                               const unsigned int nwork)
    {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= nwork)
        return;

    unsigned int i = idx + work_offset;

    unsigned int i_trial_config = blockIdx.y;
    unsigned int i_trial = (i_trial_config >> 1) % ntrial;
    unsigned int new_config = i_trial_config & 1;

    if (i >= N_local && new_config)
        return; // ghosts only exist in the old config

    // draw a Poisson variate according to the seed stored in the auxillary variable (vel.x)
    unsigned int seed_i
        = new_config ? __scalar_as_int(d_trial_vel[i].x) : __scalar_as_int(d_vel[i].x);
    hoomd::RandomGenerator rng_num(
        hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantNum, 0, 0),
        hoomd::Counter(depletant_idx(depletant_type_a, depletant_type_b), seed_i, i_trial));

    unsigned int type_i = __scalar_as_int(d_postype[i].w);
    Scalar lambda = d_lambda[type_i * depletant_idx.getNumElements()
                             + depletant_idx(depletant_type_a, depletant_type_b)];
    unsigned int n = hoomd::PoissonDistribution<Scalar>(lambda)(rng_num);

    // store result
    d_n_depletants[i * 2 * ntrial + new_config * ntrial + i_trial] = n;
    }

__global__ void hpmc_reduce_counters(const unsigned int ngpu,
                                     const unsigned int pitch,
                                     const hpmc_counters_t* d_per_device_counters,
                                     hpmc_counters_t* d_counters,
                                     const unsigned int implicit_pitch,
                                     const Index2D depletant_idx,
                                     const hpmc_implicit_counters_t* d_per_device_implicit_counters,
                                     hpmc_implicit_counters_t* d_implicit_counters)
    {
    for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
        {
        *d_counters = *d_counters + d_per_device_counters[igpu * pitch];

        for (unsigned int itype = 0; itype < depletant_idx.getNumElements(); ++itype)
            d_implicit_counters[itype]
                = d_implicit_counters[itype]
                  + d_per_device_implicit_counters[itype + igpu * implicit_pitch];
        }
    }

//! Kernel to perform the Metroplis-Hastings step for depletants
__global__ void hpmc_depletants_accept(const uint16_t seed,
                                       const uint64_t timestep,
                                       const unsigned int select,
                                       const unsigned int rank,
                                       const int* d_deltaF_int,
                                       const Index2D depletant_idx,
                                       const unsigned int deltaF_pitch,
                                       const Scalar* d_fugacity,
                                       const unsigned int* d_ntrial,
                                       unsigned int* d_reject_out,
                                       const unsigned int nwork,
                                       const unsigned work_offset)
    {
    // the particle we are handling
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= nwork)
        return;
    unsigned int i = work_idx + work_offset;

    // reduce free energy over depletant type pairs
    Scalar deltaF_i(0.0);
    for (unsigned int itype = 0; itype < depletant_idx.getW(); ++itype)
        for (unsigned int jtype = itype; jtype < depletant_idx.getH(); ++jtype)
            {
            // it is important that this loop is serial, to eliminate non-determism
            // in the acceptance loop (too much noise can make convergence difficult)
            unsigned int ntrial = d_ntrial[depletant_idx(itype, jtype)];
            Scalar fugacity = d_fugacity[depletant_idx(itype, jtype)];

            if (fugacity == 0.0 || ntrial == 0)
                continue;

            // rescale deltaF to units of kBT
            int dF_int_i = d_deltaF_int[deltaF_pitch * depletant_idx(itype, jtype) + i];
            deltaF_i += log(1 + 1 / (Scalar)ntrial) * dF_int_i;
            }

    hoomd::RandomGenerator rng_accept(
        hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantsAccept, timestep, seed),
        hoomd::Counter(i, rank, select));

    Scalar u = hoomd::UniformDistribution<Scalar>()(rng_accept);
    bool accept = u <= exp(deltaF_i);

    // update the reject flags
    if (!accept)
        atomicAdd(&d_reject_out[i], 1);
    }

    } // end namespace kernel

void generate_num_depletants(const uint16_t seed,
                             const uint64_t timestep,
                             const unsigned int select,
                             const unsigned int rank,
                             const unsigned int depletant_type_a,
                             const unsigned int depletant_type_b,
                             const Index2D depletant_idx,
                             const Scalar* d_lambda,
                             const Scalar4* d_postype,
                             unsigned int* d_n_depletants,
                             const unsigned int block_size,
                             const hipStream_t* streams,
                             const GPUPartition& gpu_partition)
    {
    // determine the maximum block size and clamp the input block size down
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::generate_num_depletants));
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);
        unsigned int nwork = range.second - range.first;

        hipLaunchKernelGGL(kernel::generate_num_depletants,
                           nwork / run_block_size + 1,
                           run_block_size,
                           0,
                           streams[idev],
                           seed,
                           timestep,
                           select,
                           rank,
                           depletant_type_a,
                           depletant_type_b,
                           depletant_idx,
                           range.first,
                           nwork,
                           d_lambda,
                           d_postype,
                           d_n_depletants);
        }
    }

void generate_num_depletants_ntrial(const Scalar4* d_vel,
                                    const Scalar4* d_trial_vel,
                                    const unsigned int ntrial,
                                    const unsigned int depletant_type_a,
                                    const unsigned int depletant_type_b,
                                    const Index2D depletant_idx,
                                    const Scalar* d_lambda,
                                    const Scalar4* d_postype,
                                    unsigned int* d_n_depletants,
                                    const unsigned int N_local,
                                    const bool add_ghosts,
                                    const unsigned int n_ghosts,
                                    const GPUPartition& gpu_partition,
                                    const unsigned int block_size,
                                    const hipStream_t* streams)
    {
    // determine the maximum block size and clamp the input block size down
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,
                         reinterpret_cast<const void*>(kernel::generate_num_depletants_ntrial));
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // add ghosts to final range
        if (idev == (int)gpu_partition.getNumActiveGPUs() - 1 && add_ghosts)
            nwork += n_ghosts;

        if (!nwork)
            continue;

        dim3 grid(nwork / run_block_size + 1, 2 * ntrial, 1);
        dim3 threads(run_block_size, 1, 1);

        hipLaunchKernelGGL((kernel::generate_num_depletants_ntrial),
                           grid,
                           threads,
                           0,
                           streams[idev],
                           d_vel,
                           d_trial_vel,
                           ntrial,
                           depletant_type_a,
                           depletant_type_b,
                           depletant_idx,
                           d_lambda,
                           d_postype,
                           d_n_depletants,
                           N_local,
                           range.first,
                           nwork);
        }
    }

void get_max_num_depletants(unsigned int* d_n_depletants,
                            unsigned int* max_n_depletants,
                            const hipStream_t* streams,
                            const GPUPartition& gpu_partition,
                            CachedAllocator& alloc)
    {
    assert(d_n_depletants);
    thrust::device_ptr<unsigned int> n_depletants(d_n_depletants);
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

#ifdef __HIP_PLATFORM_HCC__
        max_n_depletants[idev] = thrust::reduce(thrust::hip::par(alloc).on(streams[idev]),
#else
        max_n_depletants[idev] = thrust::reduce(thrust::cuda::par(alloc).on(streams[idev]),
#endif
                                                n_depletants + range.first,
                                                n_depletants + range.second,
                                                0,
                                                thrust::maximum<unsigned int>());
        }
    }

//! Compute the max # of depletants per particle, trial insertion, and configuration
void get_max_num_depletants_ntrial(const unsigned int ntrial,
                                   unsigned int* d_n_depletants,
                                   unsigned int* max_n_depletants,
                                   const bool add_ghosts,
                                   const unsigned int n_ghosts,
                                   const hipStream_t* streams,
                                   const GPUPartition& gpu_partition,
                                   CachedAllocator& alloc)
    {
    assert(d_n_depletants);
    thrust::device_ptr<unsigned int> n_depletants(d_n_depletants);
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // add ghosts to final range
        if (idev == (int)gpu_partition.getNumActiveGPUs() - 1 && add_ghosts)
            nwork += n_ghosts;

#ifdef __HIP_PLATFORM_HCC__
        max_n_depletants[idev] = thrust::reduce(thrust::hip::par(alloc).on(streams[idev]),
#else
        max_n_depletants[idev] = thrust::reduce(thrust::cuda::par(alloc).on(streams[idev]),
#endif
                                                n_depletants + range.first * 2 * ntrial,
                                                n_depletants + (range.first + nwork) * 2 * ntrial,
                                                0,
                                                thrust::maximum<unsigned int>());
        }
    }

void reduce_counters(const unsigned int ngpu,
                     const unsigned int pitch,
                     const hpmc_counters_t* d_per_device_counters,
                     hpmc_counters_t* d_counters,
                     const unsigned int implicit_pitch,
                     const Index2D depletant_idx,
                     const hpmc_implicit_counters_t* d_per_device_implicit_counters,
                     hpmc_implicit_counters_t* d_implicit_counters)
    {
    hipLaunchKernelGGL(kernel::hpmc_reduce_counters,
                       1,
                       1,
                       0,
                       0,
                       ngpu,
                       pitch,
                       d_per_device_counters,
                       d_counters,
                       implicit_pitch,
                       depletant_idx,
                       d_per_device_implicit_counters,
                       d_implicit_counters);
    }

void hpmc_depletants_accept(const uint16_t seed,
                            const uint64_t timestep,
                            const unsigned int select,
                            const unsigned int rank,
                            const int* d_deltaF_int,
                            const Index2D depletant_idx,
                            const unsigned int deltaF_pitch,
                            const Scalar* d_fugacity,
                            const unsigned int* d_ntrial,
                            unsigned int* d_reject_out,
                            const GPUPartition& gpu_partition,
                            const unsigned int block_size)
    {
    // determine the maximum block size and clamp the input block size down
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_depletants_accept));
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    assert(d_deltaF_int);
    assert(d_fugacity);
    assert(d_ntrial);
    assert(d_reject_out);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);
        unsigned int nwork = range.second - range.first;

        hipLaunchKernelGGL(kernel::hpmc_depletants_accept,
                           nwork / run_block_size + 1,
                           run_block_size,
                           0,
                           0,
                           seed,
                           timestep,
                           select,
                           rank,
                           d_deltaF_int,
                           depletant_idx,
                           deltaF_pitch,
                           d_fugacity,
                           d_ntrial,
                           d_reject_out,
                           nwork,
                           range.first);
        }
    }
    } // end namespace gpu
    } // end namespace hpmc
    } // end namespace hoomd
