// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file UpdaterClustersGPU.cu
    \brief Implements a connected components algorithm on the GPU
*/

#include <hip/hip_runtime.h>
#include "hoomd/CachedAllocator.h"

namespace hpmc
{

namespace detail
{

void gpu_connected_components(
    const uint2 *d_adj,
    unsigned int N,
    unsigned int n_elements,
    int *d_components,
    unsigned int &num_components,
    const hipDeviceProp_t& dev_prop,
    CachedAllocator& alloc);

} // end namespace detail
} // end namespace hpmc
