// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/ActiveForceConstraintCompute.h"
#include "hoomd/md/Manifold@_manifold_cpp@.h"

#define MANIFOLD_CLASS Manifold@_manifold_cpp@
#define EXPORT_FUNCTION export_ActiveForceConstraintCompute@_manifold@
// clang-format on

namespace hoomd
    {
namespace md
    {

// This should not be needed as the export_ method below instantiates the template, but runtime
// errors result without this line.
template class ActiveForceConstraintCompute<MANIFOLD_CLASS>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_ActiveForceConstraintCompute<MANIFOLD_CLASS>(m,
                                                        "ActiveForceConstraintCompute@_manifold@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
