// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoImplicitGPU.cuh"

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoImplicitGPU.cu
    \brief Definition of CUDA kernels and drivers for IntegratorHPMCMonoImplicit
*/

//! Kernel to compute the configurational bias weights
__global__ void gpu_implicit_compute_weights_kernel(unsigned int n_overlaps,
             unsigned int *d_n_success_forward,
             unsigned int *d_n_overlap_shape_forward,
             unsigned int *d_n_success_reverse,
             unsigned int *d_n_overlap_shape_reverse,
             float *d_lnb,
             unsigned int *d_n_success_zero,
             unsigned int *d_depletant_active_cell)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx >= n_overlaps)
        return;

    unsigned int n_success_forward = d_n_success_forward[idx];

    // we use float for probability
    float lnb(0.0);
    if (n_success_forward != 0)
        {
        lnb = logf((Scalar)n_success_forward/(Scalar)d_n_overlap_shape_forward[idx]);
        lnb -= logf((Scalar)d_n_success_reverse[idx]/(Scalar)d_n_overlap_shape_reverse[idx]);
        }
    else
        {
        // flag that the argument is zero
        d_n_success_zero[d_depletant_active_cell[idx]] = 1;
        }

    // write out result
    d_lnb[idx] = lnb;
    }

//! Set up cuRAND for the maximum kernel parameters
__global__ void gpu_curand_implicit_setup(unsigned int n_rng,
                                          unsigned int seed,
                                          unsigned int timestep,
                                          curandState_t *d_state)
    {
    // one active cell per thread block
    unsigned int irng = blockIdx.x*blockDim.x + threadIdx.x;

    if (irng >= n_rng) return;

    curand_init((unsigned long long)seed+(unsigned long long)irng, (unsigned long long)timestep, 0, &d_state[irng]);
    }

}; // end namespace detail

} // end namespace hpmc
