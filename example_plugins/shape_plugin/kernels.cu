// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeMySphere.h"
#include "hoomd/hpmc/ComputeFreeVolumeGPU.cuh"
#include "hoomd/hpmc/IntegratorHPMCMonoGPU.cuh"
#include "hoomd/hpmc/IntegratorHPMCMonoGPUDepletants.cuh"
#include "hoomd/hpmc/IntegratorHPMCMonoGPUDepletantsAuxilliaryPhase1.cuh"
#include "hoomd/hpmc/IntegratorHPMCMonoGPUDepletantsAuxilliaryPhase2.cuh"
#include "hoomd/hpmc/IntegratorHPMCMonoGPUMoves.cuh"
#include "hoomd/hpmc/UpdaterClustersGPU.cuh"
#include "hoomd/hpmc/UpdaterClustersGPUDepletants.cuh"

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
template hipError_t
gpu_hpmc_free_volume<ShapeMySphere>(const hpmc_free_volume_args_t& args,
                                    const typename ShapeMySphere::param_type* d_params);
    }
namespace gpu
    {
template void hpmc_gen_moves<ShapeMySphere>(const hpmc_args_t& args,
                                            const ShapeMySphere::param_type* params);

template void hpmc_narrow_phase<ShapeMySphere>(const hpmc_args_t& args,
                                               const ShapeMySphere::param_type* params);

template void hpmc_insert_depletants<ShapeMySphere>(const hpmc_args_t& args,
                                                    const hpmc_implicit_args_t& implicit_args,
                                                    const ShapeMySphere::param_type* params);

template void hpmc_update_pdata<ShapeMySphere>(const hpmc_update_args_t& args,
                                               const ShapeMySphere::param_type* params);

template void hpmc_cluster_overlaps<ShapeMySphere>(const cluster_args_t& args,
                                                   const ShapeMySphere::param_type* params);

template void hpmc_clusters_depletants<ShapeMySphere>(const cluster_args_t& args,
                                                      const hpmc_implicit_args_t& implicit_args,
                                                      const ShapeMySphere::param_type* params);

template void transform_particles<ShapeMySphere>(const clusters_transform_args_t& args,
                                                 const ShapeMySphere::param_type* params);

template void
hpmc_depletants_auxilliary_phase1<ShapeMySphere>(const hpmc_args_t& args,
                                                 const hpmc_implicit_args_t& implicit_args,
                                                 const hpmc_auxilliary_args_t& auxilliary_args,
                                                 const ShapeMySphere::param_type* params);

template void
hpmc_depletants_auxilliary_phase2<ShapeMySphere>(const hpmc_args_t& args,
                                                 const hpmc_implicit_args_t& implicit_args,
                                                 const hpmc_auxilliary_args_t& auxilliary_args,
                                                 const ShapeMySphere::param_type* params);
    } // namespace gpu

    } // end namespace hpmc
    } // end namespace hoomd
