// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPair.h"
#include "hoomd/md/EvaluatorPair@_evaluator_cpp@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialPair@_evaluator@(pybind11::module &m)
    {
    export_PotentialPair<EvaluatorPair@_evaluator_cpp@>(m, "PotentialPair@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
