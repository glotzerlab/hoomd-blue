// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"

#include "ShapeSimplePolygon.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeSimplePolygon
template cudaError_t gpu_hpmc_free_volume<ShapeSimplePolygon>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeSimplePolygon::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSimplePolygon>(const hpmc_args_t& args,
                                                  const typename ShapeSimplePolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeSimplePolygon>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSimplePolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeSimplePolygon>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSimplePolygon::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeSimplePolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSimplePolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeSimplePolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSimplePolygon::param_type *d_params);

}; // end namespace detail

} // end namespace hpmc
