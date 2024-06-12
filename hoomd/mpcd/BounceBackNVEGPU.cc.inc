// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackNVE@_geometry@GPU.cc
 * \brief Template instantation for BounceBackNVEGPU with @_geometry@.
 *
 * The geometry class is filled in by CMake using configure_file(). See
 * mpcd/CMakeLists.txt for list of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/BounceBackNVEGPU.h"

#define GEOMETRY_CLASS @_geometry@
// clang-format on

namespace hoomd
    {
namespace mpcd
    {

// Use CPU class from another compilation unit
extern template class BounceBackNVE<GEOMETRY_CLASS>;

namespace detail
    {
template void export_BounceBackNVEGPU<GEOMETRY_CLASS>(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
