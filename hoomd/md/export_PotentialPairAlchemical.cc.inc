// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPairAlchemical.h"
#include "hoomd/md/EvaluatorPair@_evaluator@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialPairAlchemical@_evaluator@(pybind11::module &m)
    {
    export_PotentialPairAlchemical<EvaluatorPair@_evaluator@>(m, "PotentialPairAlchemical@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
