// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"

#include "ShapePolyhedron.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapePolyhedron
template cudaError_t gpu_hpmc_free_volume<ShapePolyhedron>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapePolyhedron>(const hpmc_args_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapePolyhedron>(const hpmc_implicit_args_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapePolyhedron>(const hpmc_implicit_args_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapePolyhedron>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapePolyhedron>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);

}; // end namespace detail

} // end namespace hpmc
