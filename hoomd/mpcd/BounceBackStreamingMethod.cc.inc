// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackStreamingMethod@_geometry@@_force@.cc
 * \brief Template instantation for BounceBackStreamingMethod with @_geometry@ and @_force@
 *
 * The geometry and force classes are filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/BounceBackStreamingMethod.h"

#define GEOMETRY_CLASS @_geometry@
#define FORCE_CLASS @_force@
#define EXPORT_FUNCTION export_BounceBackStreamingMethod@_geometry@@_force@
// clang-format on

namespace hoomd
    {
namespace mpcd
    {

// Explicit instantiation of CPU class
template class BounceBackStreamingMethod<GEOMETRY_CLASS, FORCE_CLASS>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_BounceBackStreamingMethod<GEOMETRY_CLASS, FORCE_CLASS>(m);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
