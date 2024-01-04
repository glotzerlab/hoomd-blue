// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BulkStreamingMethodGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::BulkStreamingMethodGPU
 */

#include "BounceBackStreamingMethodGPU.cuh"
#include "BulkGeometry.h"

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {

//! Template instantiation of bulk geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<mpcd::detail::BulkGeometry>(const stream_args_t& args,
                                            const mpcd::detail::BulkGeometry& geom);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
