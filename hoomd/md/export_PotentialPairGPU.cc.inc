// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPairGPU.h"
#include "hoomd/md/EvaluatorPair@_evaluator_cpp@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialPair@_evaluator@GPU(pybind11::module &m)
    {
    export_PotentialPairGPU<EvaluatorPair@_evaluator_cpp@>(m, "PotentialPair@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
