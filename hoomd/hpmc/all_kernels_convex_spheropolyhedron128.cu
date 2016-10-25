// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"

#include "ShapeSpheropolyhedron.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeSpheropolyhedron<128>
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron<128> >(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeSpheropolyhedron<128> ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSpheropolyhedron<128> >(const hpmc_args_t& args,
                                                  const typename ShapeSpheropolyhedron<128> ::param_type *d_params);
template void gpu_hpmc_implicit_count_overlaps<ShapeSpheropolyhedron<128> >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSpheropolyhedron<128> ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeSpheropolyhedron<128> >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSpheropolyhedron<128> ::param_type *d_params);

}; // end namespace detail

} // end namespace hpmc
