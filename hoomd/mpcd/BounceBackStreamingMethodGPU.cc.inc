// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackStreamingMethod@_geometry@@_force@GPU.cc
 * \brief Template instantation for BounceBackStreamingMethodGPU with @_geometry@ and @_force@
 *
 * The geometry and force classes are filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/BounceBackStreamingMethodGPU.h"

#define GEOMETRY_CLASS @_geometry@
#define FORCE_CLASS @_force@
// clang-format on

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
template void export_BounceBackStreamingMethodGPU<GEOMETRY_CLASS, FORCE_CLASS>(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd