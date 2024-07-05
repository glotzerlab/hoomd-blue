// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ExternalField.cu
 * \brief Template device_new methods for external fields.
 *
 * \warning
 * Because __HIPCC__ seems to establish the virtual table with these device functions,
 * other device functions using the ExternalField object need to be compiled with
 * this bit using separable compilation. The consequence of all this is that
 * new ExternalFields cannot be added through the plugin interface.
 */

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

namespace hoomd
    {
template mpcd::BlockForce* hoomd::gpu::device_new(Scalar, Scalar, Scalar);
template mpcd::ConstantForce* hoomd::gpu::device_new(Scalar3);
template mpcd::SineForce* hoomd::gpu::device_new(Scalar, Scalar);
template void hoomd::gpu::device_delete(mpcd::ExternalField*);

    } // end namespace hoomd
