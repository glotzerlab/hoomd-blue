// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file LoadBalancerGPU.cu
    \brief Implementation the GPU functions for load balancing
*/

#ifdef ENABLE_MPI

#include "LoadBalancerGPU.cuh"
#include "hoomd/extern/cub/cub/cub.cuh"

//! Mark the particles that are off rank
/*!
 * \param d_ranks The current rank of each particle
 * \param d_pos Particle positions
 * \param d_cart_ranks Map from Cartesian coordinates to rank number
 * \param rank_pos Cartesian coordinates of current rank
 * \param box Local box
 * \param di Domain indexer
 * \param N Number of local particles
 *
 * Using a thread per particle, the current rank of each particle is computed assuming that a particle cannot migrate
 * more than a single rank in any direction. The Cartesian rank of the particle is computed, and mapped back to a physical
 * rank.
 */
__global__ void gpu_load_balance_mark_rank_kernel(unsigned int *d_ranks,
                                                  const Scalar4 *d_pos,
                                                  const unsigned int *d_cart_ranks,
                                                  const uint3 rank_pos,
                                                  const BoxDim box,
                                                  const Index3D di,
                                                  const unsigned int N)
    {
    // particle index
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N)
        return;

    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar3 f = box.makeFraction(pos);

    int3 grid_pos = make_int3(rank_pos.x, rank_pos.y, rank_pos.z);

    if (f.x >= Scalar(1.0)) ++grid_pos.x;
    if (f.x < Scalar(0.0)) --grid_pos.x;
    if (f.y >= Scalar(1.0)) ++grid_pos.y;
    if (f.y < Scalar(0.0)) --grid_pos.y;
    if (f.z >= Scalar(1.0)) ++grid_pos.z;
    if (f.z < Scalar(0.0)) --grid_pos.z;

    if (grid_pos.x == (int)di.getW())
        grid_pos.x = 0;
    else if (grid_pos.x < 0)
        grid_pos.x += di.getW();

    if (grid_pos.y == (int)di.getH())
        grid_pos.y = 0;
    else if (grid_pos.y < 0)
        grid_pos.y += di.getH();

    if (grid_pos.z == (int)di.getD())
        grid_pos.z = 0;
    else if (grid_pos.z < 0)
        grid_pos.z += di.getD();

    const unsigned int cur_rank = d_cart_ranks[di(grid_pos.x,grid_pos.y,grid_pos.z)];

    d_ranks[idx] = cur_rank;
    }

/*!
 * \param d_ranks The current rank of each particle
 * \param d_pos Particle positions
 * \param d_cart_ranks Map from Cartesian coordinates to rank number
 * \param rank_pos Cartesian coordinates of current rank
 * \param box Local box
 * \param di Domain indexer
 * \param N Number of local particles
 * \param block_size Kernel launch block size
 *
 * This simply a kernel driver, see gpu_load_balance_mark_rank_kernel for details.
 */
void gpu_load_balance_mark_rank(unsigned int *d_ranks,
                                const Scalar4 *d_pos,
                                const unsigned int *d_cart_ranks,
                                const uint3 rank_pos,
                                const BoxDim& box,
                                const Index3D& di,
                                const unsigned int N,
                                const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_load_balance_mark_rank_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }
    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int n_blocks = N/run_block_size + 1;

    gpu_load_balance_mark_rank_kernel<<<n_blocks, run_block_size>>>(d_ranks, d_pos, d_cart_ranks, rank_pos, box, di, N);
    }

//! Functor for selecting ranks not equal to the current rank
struct NotEqual
    {
    unsigned int not_eq_val; //!< Value to test if not equal to

    __host__ __device__ __forceinline__
    NotEqual(unsigned int _not_eq_val) : not_eq_val(_not_eq_val) {}

    __host__ __device__ __forceinline__
    bool operator()(const unsigned int &a) const
        {
        return (a != not_eq_val);
        }
    };

/*!
 * \param d_off_rank (Reduced) list of particles that are off the current rank
 * \param d_n_select Number of particles that are off the current rank
 * \param d_ranks The current rank of each particle
 * \param d_tmp_storage Temporary storage array, or NULL
 * \param tmp_storage_bytes Size of temporary storage, or 0
 * \param N Number of local particles
 * \param cur_rank Current rank index
 *
 * This function uses the CUB DeviceSelect::If primitive to select particles that are off rank using the NotEqual
 * functor. As is usual, this function must be called twice in order to perform the selection. If \a d_tmp_storage
 * is NULL, the temporary storage requirement is computed and saved in \a tmp_storage_bytes. This is externally
 * allocated from the CachedAllocator. When called the second time, the ranks of the particles not on the current
 * rank are saved in \a d_off_rank, and the number of these particles is saved in \a d_n_select.
 */
void gpu_load_balance_select_off_rank(unsigned int *d_off_rank,
                                      unsigned int *d_n_select,
                                      unsigned int *d_ranks,
                                      void *d_tmp_storage,
                                      size_t &tmp_storage_bytes,
                                      const unsigned int N,
                                      const unsigned int cur_rank)
    {
    // final precaution against calling with an empty array
    if (N == 0) return;

    NotEqual select_op(cur_rank);
    cub::DeviceSelect::If(d_tmp_storage, tmp_storage_bytes, d_ranks, d_off_rank, d_n_select, N, select_op);
    }

#endif // ENABLE_MPI
