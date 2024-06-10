// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellListGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::CellListGPU
 */

#include "CellListGPU.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {
//! Kernel to compute the MPCD cell list on the GPU
/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_list 2D array of MPCD particles in each cell
 * \param d_conditions Conditions flags for error reporting
 * \param d_vel MPCD particle velocities
 * \param d_embed_cell_ids Cell indexes of embedded particles
 * \param d_pos MPCD particle positions
 * \param d_pos_embed Particle positions
 * \param d_embed_member_idx Indexes of embedded particles in \a d_pos_embed
 * \param periodic Flags if local simulation is periodic
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_box Global simulation box
 * \param n_global_cell Global dimensions of the cell list, including padding
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_np_max Maximum number of particles per cell
 * \param cell_indexer 3D indexer for cell id
 * \param cell_list_indexer 2D indexer for particle position in cell
 * \param N_mpcd Number of MPCD particles
 * \param N_tot Total number of particle (MPCD + embedded)
 *
 * \b Implementation
 * One thread is launched per particle. The particle is floored into a bin subject to a random grid
 * shift. The number of particles in that bin is atomically incremented. If the addition of the
 * particle will not overflow the allocated memory, the particle is written into that bin.
 * Otherwise, a flag is set to resize the cell list and recompute. The MPCD particle's cell id is
 * stashed into the velocity array.
 */
__global__ void compute_cell_list(unsigned int* d_cell_np,
                                  unsigned int* d_cell_list,
                                  uint3* d_conditions,
                                  Scalar4* d_vel,
                                  unsigned int* d_embed_cell_ids,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_pos_embed,
                                  const unsigned int* d_embed_member_idx,
                                  const uchar3 periodic,
                                  const int3 origin_idx,
                                  const Scalar3 grid_shift,
                                  const BoxDim global_box,
                                  const uint3 n_global_cell,
                                  const uint3 global_cell_dim,
                                  const unsigned int cell_np_max,
                                  const Index3D cell_indexer,
                                  const Index2D cell_list_indexer,
                                  const unsigned int N_mpcd,
                                  const unsigned int N_tot)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    Scalar4 postype_i;
    if (idx < N_mpcd)
        {
        postype_i = d_pos[idx];
        }
    else
        {
        postype_i = d_pos_embed[d_embed_member_idx[idx - N_mpcd]];
        }
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);

    if (isnan(pos_i.x) || isnan(pos_i.y) || isnan(pos_i.z))
        {
        (*d_conditions).y = idx + 1;
        return;
        }

    // bin particle with grid shift
    const Scalar3 fractional_pos_i = global_box.makeFraction(pos_i - grid_shift);
    int3 global_bin = make_int3((int)std::floor(fractional_pos_i.x * global_cell_dim.x),
                                (int)std::floor(fractional_pos_i.y * global_cell_dim.y),
                                (int)std::floor(fractional_pos_i.z * global_cell_dim.z));

    // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
    // this is done using periodic from the "local" box, since this will be periodic
    // only when there is one rank along the dimension
    if (periodic.x)
        {
        if (global_bin.x == (int)n_global_cell.x)
            global_bin.x = 0;
        else if (global_bin.x == -1)
            global_bin.x = n_global_cell.x - 1;
        }
    if (periodic.y)
        {
        if (global_bin.y == (int)n_global_cell.y)
            global_bin.y = 0;
        else if (global_bin.y == -1)
            global_bin.y = n_global_cell.y - 1;
        }
    if (periodic.z)
        {
        if (global_bin.z == (int)n_global_cell.z)
            global_bin.z = 0;
        else if (global_bin.z == -1)
            global_bin.z = n_global_cell.z - 1;
        }

    // compute the local cell
    int3 bin = make_int3(global_bin.x - origin_idx.x,
                         global_bin.y - origin_idx.y,
                         global_bin.z - origin_idx.z);

    // validate and make sure no particles blew out of the box
    if ((bin.x < 0 || bin.x >= (int)cell_indexer.getW())
        || (bin.y < 0 || bin.y >= (int)cell_indexer.getH())
        || (bin.z < 0 || bin.z >= (int)cell_indexer.getD()))
        {
        (*d_conditions).z = idx + 1;
        return;
        }

    const unsigned int bin_idx = cell_indexer(bin.x, bin.y, bin.z);
    const unsigned int offset = atomicInc(&d_cell_np[bin_idx], 0xffffffff);
    if (offset < cell_np_max)
        {
        d_cell_list[cell_list_indexer(offset, bin_idx)] = idx;
        }
    else
        {
        // overflow
        atomicMax(&(*d_conditions).x, offset + 1);
        }

    // stash the current particle bin into the velocity array
    if (idx < N_mpcd)
        {
        d_vel[idx].w = __int_as_scalar(bin_idx);
        }
    else
        {
        d_embed_cell_ids[idx - N_mpcd] = bin_idx;
        }
    }

/*!
 * \param d_migrate_flag Flag signaling migration is required (output)
 * \param d_pos Embedded particle positions
 * \param d_group Indexes into \a d_pos for particles in embedded group
 * \param box Box covered by this domain
 * \param num_dim Dimensionality of system
 * \param N Number of particles in group
 *
 * \b Implementation
 * Using one thread per particle, each particle position is compared to the
 * bounds of the simulation box. If a particle lies outside the box, \a d_migrate_flag
 * has its bits set using an atomicMax transaction. The caller should then trigger
 * a communication step to migrate particles to their appropriate ranks.
 */
__global__ void cell_check_migrate_embed(unsigned int* d_migrate_flag,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_group,
                                         const BoxDim box,
                                         const unsigned int num_dim,
                                         const unsigned int N)
    {
    // one thread per particle in group
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const unsigned int idx = d_group[tid];
    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    const uchar3 periodic = box.getPeriodic();
    const Scalar3 fractional_pos = box.makeFraction(pos);
    if ((!periodic.x && (fractional_pos.x >= Scalar(1.0) || fractional_pos.x < Scalar(0.0)))
        || (!periodic.y && (fractional_pos.y >= Scalar(1.0) || fractional_pos.y < Scalar(0.0)))
        || (!periodic.z && num_dim == 3
            && (fractional_pos.z >= Scalar(1.0) || fractional_pos.z < Scalar(0.0))))
        {
        atomicMax(d_migrate_flag, 1);
        }
    }

__global__ void cell_apply_sort(unsigned int* d_cell_list,
                                const unsigned int* d_rorder,
                                const unsigned int* d_cell_np,
                                const Index2D cli,
                                const unsigned int N_mpcd,
                                const unsigned int N_cli)
    {
    // one thread per cell-list entry
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cli)
        return;

    // convert the entry 1D index into a 2D index
    const unsigned int cell = idx / cli.getW();
    const unsigned int offset = idx - (cell * cli.getW());

    /* here comes some terrible execution divergence */
    // check if the cell is filled
    const unsigned int np = d_cell_np[cell];
    if (offset < np)
        {
        // check if this is an MPCD particle
        const unsigned int pid = d_cell_list[idx];
        if (pid < N_mpcd)
            {
            d_cell_list[idx] = d_rorder[pid];
            }
        }
    }
    } // end namespace kernel
    } // end namespace gpu
    } // end namespace mpcd

/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_list 2D array of MPCD particles in each cell
 * \param d_conditions Conditions flags for error reporting
 * \param d_vel MPCD particle velocities
 * \param d_embed_cell_ids Cell indexes of embedded particles
 * \param d_pos MPCD particle positions
 * \param d_pos_embed Particle positions
 * \param d_embed_member_idx Indexes of embedded particles in \a d_pos_embed
 * \param periodic Flags if local simulation is periodic
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_box Global simulation box
 * \param n_global_cell Global dimensions of the cell list, including padding
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_np_max Maximum number of particles per cell
 * \param cell_indexer 3D indexer for cell id
 * \param cell_list_indexer 2D indexer for particle position in cell
 * \param N_mpcd Number of MPCD particles
 * \param N_tot Total number of particle (MPCD + embedded)
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion, or an error on failure
 */
cudaError_t mpcd::gpu::compute_cell_list(unsigned int* d_cell_np,
                                         unsigned int* d_cell_list,
                                         uint3* d_conditions,
                                         Scalar4* d_vel,
                                         unsigned int* d_embed_cell_ids,
                                         const Scalar4* d_pos,
                                         const Scalar4* d_pos_embed,
                                         const unsigned int* d_embed_member_idx,
                                         const uchar3& periodic,
                                         const int3& origin_idx,
                                         const Scalar3& grid_shift,
                                         const BoxDim& global_box,
                                         const uint3& n_global_cell,
                                         const uint3& global_cell_dim,
                                         const unsigned int cell_np_max,
                                         const Index3D& cell_indexer,
                                         const Index2D& cell_list_indexer,
                                         const unsigned int N_mpcd,
                                         const unsigned int N_tot,
                                         const unsigned int block_size)
    {
    // set the number of particles in each cell to zero
    cudaError_t error
        = cudaMemset(d_cell_np, 0, sizeof(unsigned int) * cell_indexer.getNumElements());
    if (error != cudaSuccess)
        return error;

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::compute_cell_list);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    mpcd::gpu::kernel::compute_cell_list<<<grid, run_block_size>>>(d_cell_np,
                                                                   d_cell_list,
                                                                   d_conditions,
                                                                   d_vel,
                                                                   d_embed_cell_ids,
                                                                   d_pos,
                                                                   d_pos_embed,
                                                                   d_embed_member_idx,
                                                                   periodic,
                                                                   origin_idx,
                                                                   grid_shift,
                                                                   global_box,
                                                                   n_global_cell,
                                                                   global_cell_dim,
                                                                   cell_np_max,
                                                                   cell_indexer,
                                                                   cell_list_indexer,
                                                                   N_mpcd,
                                                                   N_tot);

    return cudaSuccess;
    }

/*!
 * \param d_migrate_flag Flag signaling migration is required (output)
 * \param d_pos Embedded particle positions
 * \param d_group Indexes into \a d_pos for particles in embedded group
 * \param box Box covered by this domain
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::cell_check_migrate_embed
 */
cudaError_t mpcd::gpu::cell_check_migrate_embed(unsigned int* d_migrate_flag,
                                                const Scalar4* d_pos,
                                                const unsigned int* d_group,
                                                const BoxDim& box,
                                                const unsigned int num_dim,
                                                const unsigned int N,
                                                const unsigned int block_size)
    {
    // ensure that the flag is always zeroed even if the caller forgets
    cudaMemset(d_migrate_flag, 0, sizeof(unsigned int));

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::cell_check_migrate_embed);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::cell_check_migrate_embed<<<grid, run_block_size>>>(d_migrate_flag,
                                                                          d_pos,
                                                                          d_group,
                                                                          box,
                                                                          num_dim,
                                                                          N);

    return cudaSuccess;
    }

cudaError_t mpcd::gpu::cell_apply_sort(unsigned int* d_cell_list,
                                       const unsigned int* d_rorder,
                                       const unsigned int* d_cell_np,
                                       const Index2D& cli,
                                       const unsigned int N_mpcd,
                                       const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::cell_apply_sort);
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int N_cli = cli.getNumElements();

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_cli / run_block_size + 1);
    mpcd::gpu::kernel::cell_apply_sort<<<grid, run_block_size>>>(d_cell_list,
                                                                 d_rorder,
                                                                 d_cell_np,
                                                                 cli,
                                                                 N_mpcd,
                                                                 N_cli);

    return cudaSuccess;
    }

    } // end namespace hoomd
