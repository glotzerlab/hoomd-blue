// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/RejectionVirtualParticleFiller@_geometry@GPU.cu
 * \brief Template instantation for RejectionVirtualParticleFillerGPU driver (and so kernel)
 * with @_geometry@.
 *
 * The geometry class is filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/RejectionVirtualParticleFillerGPU.cuh"

#define GEOMETRY_CLASS @_geometry@
// clang-format on

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Template instantiation of slit geometry streaming
template cudaError_t __attribute__((visibility("default")))
draw_virtual_particles<GEOMETRY_CLASS>(const draw_virtual_particles_args_t& args,
                                       const GEOMETRY_CLASS& geom);
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
