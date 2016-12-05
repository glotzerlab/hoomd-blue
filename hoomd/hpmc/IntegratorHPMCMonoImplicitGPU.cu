// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoImplicitGPU.cuh"

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoImplicitGPU.cu
    \brief Definition of CUDA kernels and drivers for IntegratorHPMCMonoImplicit
*/

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
