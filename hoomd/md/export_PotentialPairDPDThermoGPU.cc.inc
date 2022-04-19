// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPairDPDThermoGPU.h"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialPairDPDThermo@_evaluator@GPU(pybind11::module &m)
    {
    export_PotentialPairDPDThermoGPU<EvaluatorPairDPDThermo@_evaluator@>(m, "PotentialPairDPDThermo@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
