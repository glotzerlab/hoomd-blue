// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ReverseNonequilibriumShearFlowGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::ReverseNonequilibriumShearFlowGPU
 */

#include "ReverseNonequilibriumShearFlowGPU.cuh"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {

__global__ void rnes_filter_particles(Scalar2* d_particles_lo,
                                      Scalar2* d_particles_hi,
                                      unsigned int* d_num_lo_hi,
                                      const Scalar4* d_pos,
                                      const Scalar4* d_vel,
                                      const Scalar mass,
                                      const Scalar2 pos_lo,
                                      const Scalar2 pos_hi,
                                      const unsigned int num_lo_alloc,
                                      const unsigned int num_hi_alloc,
                                      const unsigned int N)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 vel = d_vel[idx];
    const Scalar momentum = vel.x * mass;
    const Scalar y = d_pos[idx].y;

    if (pos_lo.x <= y && y < pos_lo.y
        && momentum > Scalar(0.0)) // lower slab, search for positive momentum
        {
        const unsigned int num_lo = atomicInc(d_num_lo_hi, 0xffffffff);
        if (num_lo < num_lo_alloc)
            {
            d_particles_lo[num_lo] = make_scalar2(__int_as_scalar(idx), momentum);
            }
        }
    else if (pos_hi.x <= y && y < pos_hi.y
             && momentum < Scalar(0.0)) // higher slab, search for negative momentum
        {
        const unsigned int num_hi = atomicInc(d_num_lo_hi + 1, 0xffffffff);
        if (num_hi < num_hi_alloc)
            {
            d_particles_hi[num_hi] = make_scalar2(__int_as_scalar(idx), momentum);
            }
        }
    }

__global__ void rnes_swap_particles(Scalar4* d_vel,
                                    Scalar* d_momentum_sum,
                                    const Scalar2* d_particles_staged,
                                    const Scalar mass,
                                    const unsigned int num_staged)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_staged)
        return;

    const Scalar2 pidx_mom = d_particles_staged[idx];
    const unsigned int pidx = __scalar_as_int(pidx_mom.x);
    const Scalar new_momentum = pidx_mom.y;

    const Scalar current_momentum = d_vel[pidx].x * mass;

    d_vel[pidx].x = new_momentum / mass;
    atomicAdd(d_momentum_sum, std::fabs(new_momentum - current_momentum));
    }

    } // end namespace kernel

cudaError_t rnes_filter_particles(Scalar2* d_particles_lo,
                                  Scalar2* d_particles_hi,
                                  unsigned int* d_num_lo_hi,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_vel,
                                  const Scalar mass,
                                  const Scalar2& pos_lo,
                                  const Scalar2& pos_hi,
                                  const unsigned int num_lo_alloc,
                                  const unsigned int num_hi_alloc,
                                  const unsigned int N,
                                  const unsigned int block_size)
    {
    // zero counts in each bin
    cudaError_t error = cudaMemset(d_num_lo_hi, 0, 2 * sizeof(unsigned int));
    if (error != cudaSuccess)
        return error;

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::rnes_filter_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::rnes_filter_particles<<<grid, run_block_size>>>(d_particles_lo,
                                                                       d_particles_hi,
                                                                       d_num_lo_hi,
                                                                       d_pos,
                                                                       d_vel,
                                                                       mass,
                                                                       pos_lo,
                                                                       pos_hi,
                                                                       num_lo_alloc,
                                                                       num_hi_alloc,
                                                                       N);

    return cudaSuccess;
    }

/*
 * thrust doesn't like to be included in host code, so explicitly list the ones
 * we need as wrappers to a template.
 */
template<class T> void rnes_thrust_sort(Scalar2* d_particles, const unsigned int N, const T& comp)
    {
    thrust::sort(thrust::device, d_particles, d_particles + N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const mpcd::detail::MinimumMomentum& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const mpcd::detail::MaximumMomentum& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const mpcd::detail::CompareMomentumToTarget& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }

cudaError_t rnes_copy_top_particles(Scalar2* h_top_particles,
                                    const Scalar2* d_particles,
                                    const unsigned int num_top)
    {
    return cudaMemcpy(h_top_particles,
                      d_particles,
                      num_top * sizeof(Scalar2),
                      cudaMemcpyDeviceToHost);
    }

cudaError_t rnes_swap_particles(Scalar4* d_vel,
                                Scalar* d_momentum_sum,
                                const Scalar2* d_particles_staged,
                                const Scalar mass,
                                const unsigned int num_staged,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::rnes_swap_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_staged / run_block_size + 1);
    mpcd::gpu::kernel::rnes_swap_particles<<<grid, run_block_size>>>(d_vel,
                                                                     d_momentum_sum,
                                                                     d_particles_staged,
                                                                     mass,
                                                                     num_staged);

    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
