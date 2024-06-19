// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_SLIT_PORE_GEOMETRY_FILLER_GPU_CUH_
#define MPCD_SLIT_PORE_GEOMETRY_FILLER_GPU_CUH_

/*!
 * \file mpcd/PlanarPoreGeometryFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::PlanarPoreGeometryFillerGPU
 */

#include <cuda_runtime.h>

#include "PlanarPoreGeometry.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Draw virtual particles in the PlanarPoreGeometry
cudaError_t slit_pore_draw_particles(Scalar4* d_pos,
                                     Scalar4* d_vel,
                                     unsigned int* d_tag,
                                     const BoxDim& box,
                                     const Scalar4* d_boxes,
                                     const uint2* d_ranges,
                                     const unsigned int num_boxes,
                                     const unsigned int N_tot,
                                     const Scalar mass,
                                     const unsigned int type,
                                     const unsigned int first_tag,
                                     const unsigned int first_idx,
                                     const Scalar kT,
                                     const uint64_t timestep,
                                     const uint16_t seed,
                                     const unsigned int block_size);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SLIT_PORE_GEOMETRY_FILLER_GPU_CUH_
