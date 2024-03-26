// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPUTypes.cuh"
#include "hoomd/GPUPartition.cuh"

namespace hoomd
    {
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

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all
   neighboring cells into the output expanded cell.
*/
__global__ void hpmc_excell(unsigned int* d_excell_idx,
                            unsigned int* d_excell_size,
                            const Index2D excli,
                            const unsigned int* d_cell_idx,
                            const unsigned int* d_cell_size,
                            const unsigned int* d_cell_adj,
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
            unsigned int neigh_cell_size = d_cell_size[neigh_cell + igpu * ci.getNumElements()];

            for (unsigned int k = 0; k < neigh_cell_size; k++)
                {
                // read in the index of the new particle to add to our cell
                unsigned int new_idx = d_cell_idx[cli(k, neigh_cell) + igpu * cli.getNumElements()];
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
__global__ void hpmc_shift(Scalar4* d_postype,
                           int3* d_image,
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
__global__ void hpmc_check_convergence(const unsigned int* d_trial_move_type,
                                       const unsigned int* d_reject_out_of_cell,
                                       unsigned int* d_reject_in,
                                       unsigned int* d_reject_out,
                                       unsigned int* d_condition,
                                       const unsigned int nwork,
                                       const unsigned work_offset)
    {
    // the particle we are handling
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= nwork)
        return;
    unsigned int i = work_idx + work_offset;

    // is this particle considered?
    bool move_active = d_trial_move_type[i] > 0;

    // combine with reject flag from gen_moves for particles which are always rejected
    bool reject_out_of_cell = d_reject_out_of_cell[i];
    bool reject = d_reject_out[i];

    // did the answer change since the last iteration?
    if (move_active && !reject_out_of_cell && reject != d_reject_in[i])
        {
        // flag that we're not done yet (a trivial race condition upon write)
        *d_condition = 1;
        }

    // update the reject flags
    d_reject_out[i] = reject || reject_out_of_cell;

    // clear input
    d_reject_in[i] = reject_out_of_cell;
    }
    } // end namespace kernel

//! Driver for kernel::hpmc_excell()
void __attribute__((visibility("default"))) hpmc_excell(unsigned int* d_excell_idx,
                                                        unsigned int* d_excell_size,
                                                        const Index2D& excli,
                                                        const unsigned int* d_cell_idx,
                                                        const unsigned int* d_cell_size,
                                                        const unsigned int* d_cell_adj,
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
    int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_excell));
    max_block_size = attr.maxThreadsPerBlock;

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);
    dim3 threads(run_block_size, 1, 1);
    dim3 grid(ci.getNumElements() / run_block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_excell,
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_excell_idx,
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
void __attribute__((visibility("default"))) hpmc_shift(Scalar4* d_postype,
                                                       int3* d_image,
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

    hipLaunchKernelGGL(kernel::hpmc_shift,
                       dim3(grid_shift),
                       dim3(threads_shift),
                       0,
                       0,
                       d_postype,
                       d_image,
                       N,
                       box,
                       shift);

    // after this kernel we return control of cuda managed memory to the host
    hipDeviceSynchronize();
    }

void __attribute__((visibility("default")))
hpmc_check_convergence(const unsigned int* d_trial_move_type,
                       const unsigned int* d_reject_out_of_cell,
                       unsigned int* d_reject_in,
                       unsigned int* d_reject_out,
                       unsigned int* d_condition,
                       const GPUPartition& gpu_partition,
                       const unsigned int block_size)
    {
    // determine the maximum block size and clamp the input block size down
    int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_check_convergence));
    max_block_size = attr.maxThreadsPerBlock;

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);

    dim3 threads(run_block_size, 1, 1);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork / run_block_size + 1;
        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL(kernel::hpmc_check_convergence,
                           grid,
                           threads,
                           0,
                           0,
                           d_trial_move_type,
                           d_reject_out_of_cell,
                           d_reject_in,
                           d_reject_out,
                           d_condition,
                           nwork,
                           range.first);
        }
    }

    } // end namespace gpu
    } // end namespace hpmc
    } // end namespace hoomd
