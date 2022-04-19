// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPairDPDThermo.h"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialPairDPDThermo@_evaluator@(pybind11::module &m)
    {
    export_PotentialPair<PotentialPair<EvaluatorPairDPDThermo@_evaluator@>>(m, "PotentialPairDPDConservative@_evaluator@");
    export_PotentialPairDPDThermo<EvaluatorPairDPDThermo@_evaluator@>(m, "PotentialPairDPDThermo@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
