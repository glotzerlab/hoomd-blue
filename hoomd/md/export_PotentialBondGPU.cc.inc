// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialBondGPU.h"
#include "hoomd/md/EvaluatorBond@_bond@.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialBond@_bond@GPU(pybind11::module &m)
    {
    export_PotentialBondGPU<EvaluatorBond@_bond@>(m, "PotentialBond@_bond@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
