// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPU.cuh"
#include "hoomd/RandomNumbers.h"

#include "hoomd/GPUPartition.cuh"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/CachedAllocator.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


namespace hpmc
{
namespace gpu
{
namespace kernel
{

//! Kernel to generate expanded cells
/*! \param d_excell_idx Output array to list the particle indices in the expanded cells
    \param d_excell_size Output array to list the number of particles in each expanded cell
    \param excli Indexer for the expanded cells
    \param d_cell_idx Particle indices in the normal cells
    \param d_cell_size Number of particles in each cell
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer
    \param cli Cell list indexer
    \param cadji Cell adjacency indexer
    \param ngpu Number of active devices

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all neighboring cells
    into the output expanded cell.
*/
__global__ void hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D ci,
                            const Index2D cli,
                            const Index2D cadji,
                            const unsigned int ngpu)
    {
    // compute the output cell
    unsigned int my_cell = 0;
    my_cell = blockDim.x * blockIdx.x + threadIdx.x;

    if (my_cell >= ci.getNumElements())
        return;

    unsigned int my_cell_size = 0;

    // loop over neighboring cells and build up the expanded cell list
    for (unsigned int offset = 0; offset < cadji.getW(); offset++)
        {
        unsigned int neigh_cell = d_cell_adj[cadji(offset, my_cell)];

        // iterate over per-device cell lists
        for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
            {
            unsigned int neigh_cell_size = d_cell_size[neigh_cell+igpu*ci.getNumElements()];

            for (unsigned int k = 0; k < neigh_cell_size; k++)
                {
                // read in the index of the new particle to add to our cell
                unsigned int new_idx = d_cell_idx[cli(k, neigh_cell)+igpu*cli.getNumElements()];
                d_excell_idx[excli(my_cell_size, my_cell)] = new_idx;
                my_cell_size++;
                }
            }
        }

    // write out the final size
    d_excell_size[my_cell] = my_cell_size;
    }

//! Kernel for grid shift
/*! \param d_postype postype of each particle
    \param d_image Image flags for each particle
    \param N number of particles
    \param box Simulation box
    \param shift Vector by which to translate the particles

    Shift all the particles by a given vector.

    \ingroup hpmc_kernels
*/
__global__ void hpmc_shift(Scalar4 *d_postype,
                          int3 *d_image,
                          const unsigned int N,
                          const BoxDim box,
                          const Scalar3 shift)
    {
    // identify the active cell that this thread handles
    unsigned int my_pidx = blockIdx.x * blockDim.x + threadIdx.x;

    // this thread is inactive if it indexes past the end of the particle list
    if (my_pidx >= N)
        return;

    // pull in the current position
    Scalar4 postype = d_postype[my_pidx];

    // shift the position
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    pos += shift;

    // wrap the particle back into the box
    int3 image = d_image[my_pidx];
    box.wrap(pos, image);

    // write out the new position and orientation
    d_postype[my_pidx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    d_image[my_pidx] = image;
    }

//!< Kernel to evaluate convergence
__global__ void hpmc_check_convergence(
                 const unsigned int *d_trial_move_type,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject_in,
                 unsigned int *d_reject_out,
                 unsigned int *d_condition,
                 const unsigned int nwork,
                 const unsigned work_offset)
    {
    // the particle we are handling
    unsigned int work_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (work_idx >= nwork)
        return;
    unsigned int i = work_idx + work_offset;

    // is this particle considered?
    bool move_active = d_trial_move_type[i] > 0;

    // combine with reject flag from gen_moves for particles which are always rejected
    bool reject = d_reject_out_of_cell[i] || d_reject_out[i];

    // did the answer change since the last iteration?
    if (move_active && reject != d_reject_in[i])
        {
        // flag that we're not done yet (a trivial race condition upon write)
        *d_condition = 1;
        }

    // update the reject flags
    d_reject_out[i] = reject;

    // clear input
    d_reject_in[i] = 0;
    }

//! Generate number of depletants per particle
__global__ void generate_num_depletants(const unsigned int seed,
                                        const unsigned int timestep,
                                        const unsigned int select,
                                        const unsigned int num_types,
                                        const unsigned int depletant_type_a,
                                        const unsigned int depletant_type_b,
                                        const Index2D depletant_idx,
                                        const unsigned int work_offset,
                                        const unsigned int nwork,
                                        const Scalar *d_lambda,
                                        const Scalar4 *d_postype,
                                        unsigned int *d_n_depletants)
    {
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= nwork)
        return;

    idx += work_offset;

    hoomd::RandomGenerator rng_poisson(hoomd::RNGIdentifier::HPMCDepletantNum, idx, seed, timestep,
        select*depletant_idx.getNumElements() + depletant_idx(depletant_type_a,depletant_type_b));
    Index2D typpair_idx(num_types);
    unsigned int type_i = __scalar_as_int(d_postype[idx].w);
    d_n_depletants[idx] = hoomd::PoissonDistribution<Scalar>(
        d_lambda[type_i*depletant_idx.getNumElements()+depletant_idx(depletant_type_a,depletant_type_b)])(rng_poisson);
    }

__global__ void hpmc_reduce_counters(const unsigned int ngpu,
                     const unsigned int pitch,
                     const hpmc_counters_t *d_per_device_counters,
                     hpmc_counters_t *d_counters,
                     const unsigned int implicit_pitch,
                     const Index2D depletant_idx,
                     const hpmc_implicit_counters_t *d_per_device_implicit_counters,
                     hpmc_implicit_counters_t *d_implicit_counters)
    {
    for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
        {
        *d_counters = *d_counters + d_per_device_counters[igpu*pitch];

        for (unsigned int itype = 0; itype < depletant_idx.getNumElements(); ++itype)
            d_implicit_counters[itype] = d_implicit_counters[itype] + d_per_device_implicit_counters[itype+igpu*implicit_pitch];
        }
    }

} // end namespace kernel

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int *d_excell_idx,
                 unsigned int *d_excell_size,
                 const Index2D& excli,
                 const unsigned int *d_cell_idx,
                 const unsigned int *d_cell_size,
                 const unsigned int *d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size)
    {
    assert(d_excell_idx);
    assert(d_excell_size);
    assert(d_cell_idx);
    assert(d_cell_size);
    assert(d_cell_adj);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_excell));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    dim3 threads(min(block_size, (unsigned int)max_block_size), 1, 1);
    dim3 grid(ci.getNumElements() / block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_excell, dim3(grid), dim3(threads), 0, 0, d_excell_idx,
                                           d_excell_size,
                                           excli,
                                           d_cell_idx,
                                           d_cell_size,
                                           d_cell_adj,
                                           ci,
                                           cli,
                                           cadji,
                                           ngpu);

    }

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
                int3 *d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size)
    {
    assert(d_postype);
    assert(d_image);

    // setup the grid to run the kernel
    dim3 threads_shift(block_size, 1, 1);
    dim3 grid_shift(N / block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_shift, dim3(grid_shift), dim3(threads_shift), 0, 0, d_postype,
                                                      d_image,
                                                      N,
                                                      box,
                                                      shift);

    // after this kernel we return control of cuda managed memory to the host
    hipDeviceSynchronize();
    }


void hpmc_check_convergence(const unsigned int *d_trial_move_type,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject_in,
                 unsigned int *d_reject_out,
                 unsigned int *d_condition,
                 const GPUPartition& gpu_partition,
                 const unsigned int block_size)
    {
    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_check_convergence));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);

    dim3 threads(run_block_size, 1, 1);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork/run_block_size + 1;
        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL(kernel::hpmc_check_convergence, grid, threads, 0, 0,
            d_trial_move_type,
            d_reject_out_of_cell,
            d_reject_in,
            d_reject_out,
            d_condition,
            nwork,
            range.first);
        }
    }

void generate_num_depletants(const unsigned int seed,
                             const unsigned int timestep,
                             const unsigned int select,
                             const unsigned int num_types,
                             const unsigned int depletant_type_a,
                             const unsigned int depletant_type_b,
                             const Index2D depletant_idx,
                             const Scalar *d_lambda,
                             const Scalar4 *d_postype,
                             unsigned int *d_n_depletants,
                             const unsigned int block_size,
                             const hipStream_t *streams,
                             const GPUPartition& gpu_partition)
    {
    // determine the maximum block size and clamp the input block size down
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::generate_num_depletants));
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);
        unsigned int nwork = range.second - range.first;

        hipLaunchKernelGGL(kernel::generate_num_depletants, nwork/run_block_size+1, run_block_size, 0, streams[idev],
            seed,
            timestep,
            select,
            num_types,
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

void get_max_num_depletants(unsigned int *d_n_depletants,
                            unsigned int *max_n_depletants,
                            const hipStream_t *streams,
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

void reduce_counters(const unsigned int ngpu,
                     const unsigned int pitch,
                     const hpmc_counters_t *d_per_device_counters,
                     hpmc_counters_t *d_counters,
                     const unsigned int implicit_pitch,
                     const Index2D depletant_idx,
                     const hpmc_implicit_counters_t *d_per_device_implicit_counters,
                     hpmc_implicit_counters_t *d_implicit_counters)
    {
    hipLaunchKernelGGL(kernel::hpmc_reduce_counters, 1, 1, 0, 0,
                     ngpu,
                     pitch,
                     d_per_device_counters,
                     d_counters,
                     implicit_pitch,
                     depletant_idx,
                     d_per_device_implicit_counters,
                     d_implicit_counters);
    }

} // end namespace gpu
} // end namespace hpmc

