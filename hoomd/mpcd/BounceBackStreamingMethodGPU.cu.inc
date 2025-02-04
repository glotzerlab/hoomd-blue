// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackStreamingMethod@_geometry@@_force@GPU.cu
 * \brief Template instantation for BounceBackStreamingMethodGPU driver (and so kernel) with
 * @_geometry@ and @_force@.
 *
 * The geometry and force classes are filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/BounceBackStreamingMethodGPU.cuh"

#define GEOMETRY_CLASS @_geometry@
#define FORCE_CLASS @_force@
// clang-format on

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Template instantiation of bulk geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<GEOMETRY_CLASS, FORCE_CLASS>(const stream_args_t& args,
                                             const GEOMETRY_CLASS& geom,
                                             const FORCE_CLASS& force);
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
