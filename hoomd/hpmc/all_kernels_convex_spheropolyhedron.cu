// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "UpdaterClustersGPU.cuh"

#include "ShapeSpheropolyhedron.h"

namespace hpmc
{

namespace detail
{
//! HPMC kernels for ShapeSpheropolyhedron
template hipError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron >(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeSpheropolyhedron ::param_type *d_params);
}

namespace gpu
{
//! Driver for kernel::hpmc_gen_moves()
template void hpmc_gen_moves<ShapeSpheropolyhedron>(const hpmc_args_t& args, const ShapeSpheropolyhedron::param_type *params);
//! Driver for kernel::hpmc_narrow_phase()
template void hpmc_narrow_phase<ShapeSpheropolyhedron>(const hpmc_args_t& args, const ShapeSpheropolyhedron::param_type *params);
//! Driver for kernel::hpmc_insert_depletants()
template void hpmc_insert_depletants<ShapeSpheropolyhedron>(const hpmc_args_t& args, const hpmc_implicit_args_t& implicit_args, const ShapeSpheropolyhedron::param_type *params);
//! Driver for kernel::hpmc_update_pdata()
template void hpmc_update_pdata<ShapeSpheropolyhedron>(const hpmc_update_args_t& args, const ShapeSpheropolyhedron::param_type *params);

//! Kernel driver for kernel::cluster_overlaps
template void hpmc_cluster_overlaps<ShapeSpheropolyhedron>(const cluster_args_t& args, const ShapeSpheropolyhedron::param_type *params);
//! Kernel driver for kernel::clusters_depletants
template void hpmc_clusters_depletants<ShapeSpheropolyhedron>(const cluster_args_t& args, const hpmc_implicit_args_t& implicit_args, const ShapeSpheropolyhedron::param_type *params);
//! Driver for kernel::transform_particles
template void transform_particles<ShapeSpheropolyhedron>(const clusters_transform_args_t& args, const ShapeSpheropolyhedron::param_type *params);
}

} // end namespace hpmc
