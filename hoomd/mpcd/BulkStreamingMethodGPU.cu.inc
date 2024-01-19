// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BulkStreamingMethod@_force@GPU.cu
 * \brief Template instantation for BulkStreamingMethodGPU driver (and so kernel) with @_force@
 *
 * The force class is filled in by CMake using configure_file(). See mpcd/CMakeLists.txt for list
 * of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/BulkGeometry.h"
#include "hoomd/mpcd/BounceBackStreamingMethodGPU.cuh"
// clang-format on

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Template instantiation of bulk geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<detail::BulkGeometry, @_force @>(const stream_args_t& args,
                                                 const detail::BulkGeometry& geom,
                                                 const @_force @& force);
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
