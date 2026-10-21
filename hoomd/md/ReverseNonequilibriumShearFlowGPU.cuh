// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_CUH_
#define MD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_CUH_

/*!
 * \file md/ReverseNonequilibriumShearFlowGPU.cuh
 * \brief Declaration of CUDA kernels for md::ReverseNonequilibriumShearFlowGPU
 */

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"

#include "ReverseNonequilibriumShearFlowUtilities.h"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace hoomd
    {
namespace md
    {
namespace gpu
    {

//! Filter particles into slabs
template<class IndexReaderT, class VelocityMassReaderT>
cudaError_t rnes_filter_particles(Scalar2* d_particles_lo,
                                  Scalar2* d_particles_hi,
                                  unsigned int* d_num_lo_hi,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_vel,
                                  const IndexReaderT& index_reader,
                                  const VelocityMassReaderT& reader,
                                  const Scalar2& pos_lo,
                                  const Scalar2& pos_hi,
                                  const unsigned int num_lo_alloc,
                                  const unsigned int num_hi_alloc,
                                  const unsigned int N,
                                  const unsigned int block_size);

void rnes_sort(Scalar2* d_particles, const unsigned int N, const detail::MinimumMomentum& comp);
void rnes_sort(Scalar2* d_particles, const unsigned int N, const detail::MaximumMomentum& comp);
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const detail::CompareMomentumToTarget& comp);

cudaError_t rnes_copy_top_particles(Scalar2* h_top_particles,
                                    const Scalar2* d_particles,
                                    const unsigned int num_top);

template<class VelocityMassReaderT>
cudaError_t rnes_swap_particles(Scalar4* d_vel,
                                Scalar* d_momentum_sum,
                                const Scalar2* d_particles_staged,
                                const VelocityMassReaderT& reader,
                                const unsigned int num_staged,
                                const unsigned int block_size);

#ifdef __NVCC__
namespace kernel
    {

template<class IndexReaderT, class VelocityMassReaderT>
__global__ void rnes_filter_particles(Scalar2* d_particles_lo,
                                      Scalar2* d_particles_hi,
                                      unsigned int* d_num_lo_hi,
                                      const Scalar4* d_pos,
                                      const Scalar4* d_vel,
                                      const IndexReaderT index_reader,
                                      const VelocityMassReaderT reader,
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
    const unsigned int pidx = index_reader(idx);
    Scalar mass;
    Scalar3 velocity;
    reader.read(velocity, mass, pidx);

    const Scalar momentum = velocity.x * mass;
    const Scalar y = d_pos[pidx].y;

    if (pos_lo.x <= y && y < pos_lo.y
        && momentum > Scalar(0.0)) // lower slab, search for positive momentum
        {
        const unsigned int num_lo = atomicInc(d_num_lo_hi, 0xffffffff);
        if (num_lo < num_lo_alloc)
            {
            d_particles_lo[num_lo] = make_scalar2(__int_as_scalar(pidx), momentum);
            }
        }
    else if (pos_hi.x <= y && y < pos_hi.y
             && momentum < Scalar(0.0)) // higher slab, search for negative momentum
        {
        const unsigned int num_hi = atomicInc(d_num_lo_hi + 1, 0xffffffff);
        if (num_hi < num_hi_alloc)
            {
            d_particles_hi[num_hi] = make_scalar2(__int_as_scalar(pidx), momentum);
            }
        }
    }

template<class VelocityMassReaderT>
__global__ void rnes_swap_particles(Scalar4* d_vel,
                                    Scalar* d_momentum_sum,
                                    const Scalar2* d_particles_staged,
                                    const VelocityMassReaderT reader,
                                    const unsigned int num_staged)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_staged)
        return;

    const Scalar2 pidx_mom = d_particles_staged[idx];
    const unsigned int pidx = __scalar_as_int(pidx_mom.x);
    Scalar mass;
    Scalar3 velocity;
    reader.read(velocity, mass, pidx);

    const Scalar new_momentum = pidx_mom.y;

    const Scalar current_momentum = velocity.x * mass;

    d_vel[pidx].x = new_momentum / mass;
    atomicAdd(d_momentum_sum, std::fabs(new_momentum - current_momentum));
    }

    } // end namespace kernel

template<class IndexReaderT, class VelocityMassReaderT>
cudaError_t rnes_filter_particles(Scalar2* d_particles_lo,
                                  Scalar2* d_particles_hi,
                                  unsigned int* d_num_lo_hi,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_vel,
                                  const IndexReaderT& index_reader,
                                  const VelocityMassReaderT& reader,
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
    cudaFuncGetAttributes(
        &attr,
        (const void*)kernel::rnes_filter_particles<IndexReaderT, VelocityMassReaderT>);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N / run_block_size + 1);
    md::gpu::kernel::rnes_filter_particles<IndexReaderT, VelocityMassReaderT>
        <<<grid, run_block_size>>>(d_particles_lo,
                                   d_particles_hi,
                                   d_num_lo_hi,
                                   d_pos,
                                   d_vel,
                                   index_reader,
                                   reader,
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
void rnes_sort(Scalar2* d_particles, const unsigned int N, const md::detail::MinimumMomentum& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }
void rnes_sort(Scalar2* d_particles, const unsigned int N, const md::detail::MaximumMomentum& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const md::detail::CompareMomentumToTarget& comp)
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

template<class VelocityMassReaderT>
cudaError_t rnes_swap_particles(Scalar4* d_vel,
                                Scalar* d_momentum_sum,
                                const Scalar2* d_particles_staged,
                                const VelocityMassReaderT& reader,
                                const unsigned int num_staged,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::rnes_swap_particles<VelocityMassReaderT>);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_staged / run_block_size + 1);
    md::gpu::kernel::rnes_swap_particles<VelocityMassReaderT>
        <<<grid, run_block_size>>>(d_vel, d_momentum_sum, d_particles_staged, reader, num_staged);

    return cudaSuccess;
    }
#endif

    } // end namespace gpu
    } // namespace md
    } // end namespace hoomd
#endif // MD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_CUH_
