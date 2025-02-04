// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/TwoStepRATTLE@_method@.h"
#include "hoomd/md/Manifold@_manifold_cpp@.h"

#define MANIFOLD_CLASS Manifold@_manifold_cpp@
#define BASE TwoStepRATTLE@_method@
#define PYBIND_EXPORT export_TwoStepRATTLE@_method@
#define EXPORT_FUNCTION export_TwoStepRATTLE@_method@@_manifold@
// clang-format on

namespace hoomd
    {
namespace md
    {
template class BASE<MANIFOLD_CLASS>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    PYBIND_EXPORT<MANIFOLD_CLASS>(m, "TwoStepRATTLE@_method@@_manifold@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
