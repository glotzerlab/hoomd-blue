// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"

#include "ShapeConvexPolyhedron.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeConvexPolyhedron
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron >(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron >(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeConvexPolyhedron >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeConvexPolyhedron >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeConvexPolyhedron >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeConvexPolyhedron >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);


}; // end namespace detail

} // end namespace hpmc
