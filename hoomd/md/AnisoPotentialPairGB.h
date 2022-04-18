// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "AnisoPotentialPair.h"
#include "EvaluatorPairGB.h"

#ifdef ENABLE_HIP
#include "AnisoPotentialPairGPU.h"
#endif

namespace hoomd { namespace md {

extern template class AnisoPotentialPair<EvaluatorPairGB>;

#ifdef ENABLE_HIP
extern template class AnisoPotentialPairGPU<EvaluatorPairGB>;
#endif

namespace detail {
extern template void export_AnisoPotentialPair<EvaluatorPairGB>(pybind11::module& m, const std::string& name);

#ifdef ENABLE_HIP
extern template void export_AnisoPotentialPairGPU<EvaluatorPairGB>(pybind11::module& m, const std::string& name);
#endif
    }

    } // end namespace md
    } // end namespace hoomd
