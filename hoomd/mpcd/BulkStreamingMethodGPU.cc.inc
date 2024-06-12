// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BulkStreamingMethod@_force@GPU.cc
 * \brief Template instantation for BulkStreamingMethodGPU with @_force@
 *
 * The force class is filled in by CMake using configure_file(). See mpcd/CMakeLists.txt for list
 * of values that are used.
 */

// clang-format off
#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/BulkStreamingMethod.h"
#include "hoomd/mpcd/BulkStreamingMethodGPU.h"

#define FORCE_CLASS @_force@
#define EXPORT_FUNCTION export_BulkStreamingMethod@_force@GPU
// clang-format on

namespace hoomd
    {
namespace mpcd
    {

// Use CPU class from another compilation unit
extern template class BulkStreamingMethod<FORCE_CLASS>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_BulkStreamingMethodGPU<FORCE_CLASS>(m);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
