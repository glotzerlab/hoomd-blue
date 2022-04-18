// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialTersoffGPU.h"
#include "hoomd/md/Evaluator@_evaluator@.h"

namespace hoomd { namespace md { namespace detail {

void export_Potential@_evaluator@GPU(pybind11::module &m)
    {
    export_PotentialTersoffGPU<Evaluator@_evaluator@>(m, "Potential@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
