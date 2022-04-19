// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialExternalGPU.h"
#include "hoomd/md/EvaluatorExternal@_evaluator@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialExternal@_evaluator@GPU(pybind11::module &m)
    {
    export_PotentialExternalGPU<EvaluatorExternal@_evaluator@>(m, "PotentialExternal@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
