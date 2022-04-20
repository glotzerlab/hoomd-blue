// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGB.h"

namespace hoomd
    {
namespace md
    {
template class AnisoPotentialPairGPU<EvaluatorPairGB>;

namespace detail
    {
template void export_AnisoPotentialPairGPU<EvaluatorPairGB>(pybind11::module& m,
                                                            const std::string& name);

void export_AnisoPotentialPairGBGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairGB>(m, "AnisoPotentialPairGBGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
