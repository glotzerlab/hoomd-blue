// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/RejectionVirtualParticleFiller@_geometry@GPU.cc
 * \brief Template instantation for RejectionVirtualParticleFillerGPU with @_geometry@.
 *
 * The geometry class is filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/RejectionVirtualParticleFillerGPU.h"

#define GEOMETRY_CLASS @_geometry@
#define EXPORT_FUNCTION export_RejectionVirtualParticleFiller@_geometry@GPU
// clang-format on

namespace hoomd
    {
namespace mpcd
    {

// Use CPU class from another compilation unit
extern template class RejectionVirtualParticleFiller<GEOMETRY_CLASS>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_RejectionVirtualParticleFillerGPU<GEOMETRY_CLASS>(m);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
