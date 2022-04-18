// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "AnisoPotentialPair.h"
#include "EvaluatorPairALJ.h"

#ifdef ENABLE_HIP
#include "AnisoPotentialPairGPU.h"
#endif

namespace hoomd { namespace md {

extern template class AnisoPotentialPair<EvaluatorPairALJ<3>>;

#ifdef ENABLE_HIP
extern template class AnisoPotentialPairGPU<EvaluatorPairALJ<3>>;
#endif

namespace detail {
extern template void export_AnisoPotentialPair<EvaluatorPairALJ<3>>(pybind11::module& m, const std::string& name);

#ifdef ENABLE_HIP
extern template void export_AnisoPotentialPairGPU<EvaluatorPairALJ<3>>(pybind11::module& m, const std::string& name);
#endif
    }

    } // end namespace md
    } // end namespace hoomd
