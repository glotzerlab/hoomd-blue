// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ExternalField.cu
 * \brief Template device_new methods for external fields.
 */

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

template mpcd::ConstantForce* hoomd::gpu::device_new(Scalar3);
template mpcd::SineForce* hoomd::gpu::device_new(Scalar,Scalar);
template void hoomd::gpu::device_delete(mpcd::ExternalField*);
