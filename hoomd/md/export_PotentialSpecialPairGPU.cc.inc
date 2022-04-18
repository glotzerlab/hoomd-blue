// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialSpecialPairGPU.h"
#include "hoomd/md/EvaluatorSpecialPair@_pair@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialSpecialPair@_pair@GPU(pybind11::module &m)
    {
    export_PotentialSpecialPairGPU<EvaluatorSpecialPair@_pair@>(m, "PotentialSpecialPair@_pair@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
