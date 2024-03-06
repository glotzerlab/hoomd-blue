// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LoadBalancerGPU.cu
    \brief Implementation the GPU functions for load balancing
*/

#ifdef ENABLE_MPI
#include <hip/hip_runtime.h>

#include "LoadBalancerGPU.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#pragma GCC diagnostic pop

namespace hoomd
    {
namespace kernel
    {
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
 * Using a thread per particle, the current rank of each particle is computed assuming that a
 * particle cannot migrate more than a single rank in any direction. The Cartesian rank of the
 * particle is computed, and mapped back to a physical rank.
 */
__global__ void gpu_load_balance_mark_rank_kernel(unsigned int* d_ranks,
                                                  const Scalar4* d_pos,
                                                  const unsigned int* d_cart_ranks,
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

    if (f.x >= Scalar(1.0))
        ++grid_pos.x;
    if (f.x < Scalar(0.0))
        --grid_pos.x;
    if (f.y >= Scalar(1.0))
        ++grid_pos.y;
    if (f.y < Scalar(0.0))
        --grid_pos.y;
    if (f.z >= Scalar(1.0))
        ++grid_pos.z;
    if (f.z < Scalar(0.0))
        --grid_pos.z;

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

    const unsigned int cur_rank = d_cart_ranks[di(grid_pos.x, grid_pos.y, grid_pos.z)];

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
void gpu_load_balance_mark_rank(unsigned int* d_ranks,
                                const Scalar4* d_pos,
                                const unsigned int* d_cart_ranks,
                                const uint3 rank_pos,
                                const BoxDim& box,
                                const Index3D& di,
                                const unsigned int N,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_load_balance_mark_rank_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int n_blocks = N / run_block_size + 1;

    hipLaunchKernelGGL(gpu_load_balance_mark_rank_kernel,
                       dim3(n_blocks),
                       dim3(run_block_size),
                       0,
                       0,
                       d_ranks,
                       d_pos,
                       d_cart_ranks,
                       rank_pos,
                       box,
                       di,
                       N);
    }

//! Functor for selecting ranks not equal to the current rank
struct NotEqual
    {
    unsigned int not_eq_val; //!< Value to test if not equal to

    __host__ __device__ __forceinline__ NotEqual(unsigned int _not_eq_val) : not_eq_val(_not_eq_val)
        {
        }

    __host__ __device__ __forceinline__ bool operator()(const unsigned int& a) const
        {
        return (a != not_eq_val);
        }
    };

/*!
 * \param d_off_rank (Reduced) list of particles that are off the current rank
 * \param d_ranks The current rank of each particle
 * \param N Number of local particles
 * \param cur_rank Current rank index
 *
 * \returns The number of particles that are off the current rank.
 *
 * This function uses thrust::copy_if to select particles that are off rank using the NotEqual
 * functor.
 *
 * \b Note
 * This function previously used cub::DeviceSelect::If to perform this operation. But, I ran into
 * issues with that in mpcd/SorterGPU.cu. As a precaution, I am also replacing this method here.
 */
unsigned int gpu_load_balance_select_off_rank(unsigned int* d_off_rank,
                                              unsigned int* d_ranks,
                                              const unsigned int N,
                                              const unsigned int cur_rank)
    {
    // final precaution against calling with an empty array
    if (N == 0)
        return 0;

    unsigned int* last
        = thrust::copy_if(thrust::device, d_ranks, d_ranks + N, d_off_rank, NotEqual(cur_rank));
    return (unsigned int)(last - d_off_rank);
    }

    } // end namespace kernel

    } // end namespace hoomd

#endif // ENABLE_MPI
