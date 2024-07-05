// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __SFC_PACK_UPDATER_GPU_CUH__
#define __SFC_PACK_UPDATER_GPU_CUH__

#include "BoxDim.h"
#include "CachedAllocator.h"
#include "HOOMDMath.h"

/*! \file SFCPackTunerGPU.cuh
    \brief Defines GPU functions for generating the space-filling curve sorted order on the GPU.
   Used by SFCPackTunerGPU.
*/

namespace hoomd
    {
namespace kernel
    {
//! Generate sorted order on GPU
void gpu_generate_sorted_order(unsigned int N,
                               const Scalar4* d_pos,
                               unsigned int* d_particle_bins,
                               unsigned int* d_traversal_order,
                               unsigned int n_grid,
                               unsigned int* d_sorted_order,
                               const BoxDim& box,
                               bool twod,
                               CachedAllocator& alloc);

//! Reorder particle data (GPU driver function)
void gpu_apply_sorted_order(unsigned int N,
                            unsigned int n_ghost,
                            const unsigned int* d_sorted_order,
                            const Scalar4* d_pos,
                            Scalar4* d_pos_alt,
                            const Scalar4* d_vel,
                            Scalar4* d_vel_alt,
                            const Scalar3* d_accel,
                            Scalar3* d_accel_alt,
                            const Scalar* d_charge,
                            Scalar* d_charge_alt,
                            const Scalar* d_diameter,
                            Scalar* d_diameter_alt,
                            const int3* d_image,
                            int3* d_image_alt,
                            const unsigned int* d_body,
                            unsigned int* d_body_alt,
                            const unsigned int* d_tag,
                            unsigned int* d_tag_alt,
                            const Scalar4* d_orientation,
                            Scalar4* d_orientation_alt,
                            const Scalar4* d_angmom,
                            Scalar4* d_angmom_alt,
                            const Scalar3* d_inertia,
                            Scalar3* d_inertia_alt,
                            const Scalar* d_net_virial,
                            Scalar* d_net_virial_alt,
                            size_t virial_pitch,
                            const Scalar4* d_net_force,
                            Scalar4* d_net_force_alt,
                            const Scalar4* d_net_torque,
                            Scalar4* d_net_torque_alt,
                            unsigned int* d_rtag);

    } // namespace kernel

    } // end namespace hoomd

#endif // __SFC_PACK_UPDATER_GPU_CUH__
