// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/TwoStepRATTLE@_method@GPU.h"
#include "hoomd/md/Manifold@_manifold_cpp@.h"

namespace hoomd { namespace md { namespace detail {

void export_TwoStepRATTLE@_method@GPU@_manifold@(pybind11::module &m)
    {
    export_TwoStepRATTLE@_method@GPU<Manifold@_manifold_cpp@>(m,
                                                           "TwoStepRATTLE@_method@@_manifold@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
