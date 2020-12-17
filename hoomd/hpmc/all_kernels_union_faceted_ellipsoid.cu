// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"

#include "ShapeFacetedEllipsoid.h"
#include "ShapeUnion.h"

namespace hpmc
{

namespace detail
{
//! HPMC kernels for ShapeUnion<ShapeFacetedEllipsoid>
template hipError_t gpu_hpmc_free_volume<ShapeUnion<ShapeFacetedEllipsoid> >(const hpmc_free_volume_args_t &args,
                                                  const typename ShapeUnion<ShapeFacetedEllipsoid> ::param_type *d_params);
}

namespace gpu
{
//! Driver for kernel::hpmc_gen_moves()
template void hpmc_gen_moves<ShapeUnion<ShapeFacetedEllipsoid> >(const hpmc_args_t& args, const ShapeUnion<ShapeFacetedEllipsoid>::param_type *params);
//! Driver for kernel::hpmc_narrow_phase()
template void hpmc_narrow_phase<ShapeUnion<ShapeFacetedEllipsoid> >(const hpmc_args_t& args, const ShapeUnion<ShapeFacetedEllipsoid>::param_type *params);
//! Driver for kernel::hpmc_insert_depletants()
template void hpmc_insert_depletants<ShapeUnion<ShapeFacetedEllipsoid> >(const hpmc_args_t& args, const hpmc_implicit_args_t& implicit_args, const ShapeUnion<ShapeFacetedEllipsoid>::param_type *params);
//! Driver for kernel::hpmc_update_pdata()
template void hpmc_update_pdata<ShapeUnion<ShapeFacetedEllipsoid> >(const hpmc_update_args_t& args, const ShapeUnion<ShapeFacetedEllipsoid> ::param_type *params);
}

} // end namespace hpmc
