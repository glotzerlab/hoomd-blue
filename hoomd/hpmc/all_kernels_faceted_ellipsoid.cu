// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"

#include "ShapeFacetedEllipsoid.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeFacetedEllipsoid
template cudaError_t gpu_hpmc_free_volume<ShapeFacetedEllipsoid>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeFacetedEllipsoid::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeFacetedEllipsoid>(const hpmc_args_t& args,
                                                  const typename ShapeFacetedEllipsoid::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeFacetedEllipsoid>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeFacetedEllipsoid::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeFacetedEllipsoid>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeFacetedEllipsoid::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeFacetedEllipsoid>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeFacetedEllipsoid::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeFacetedEllipsoid>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeFacetedEllipsoid::param_type *d_params);

}; // end namespace detail

} // end namespace hpmc
