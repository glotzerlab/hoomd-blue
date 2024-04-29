// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairChain.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairChainGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairChain>(m, "AnisoPotentialPairChainGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
