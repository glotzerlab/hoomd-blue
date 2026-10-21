// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ReverseNonequilibriumShearFlowGPU.h"
#include "hoomd/ParticleLoader.h"

namespace hoomd
    {
namespace md
    {
template class ReverseNonequilibriumShearFlowGPU<ParticleLoader>;

namespace detail
    {
void export_ReverseNonequilibriumShearFlowGPU(pybind11::module& m)
    {
    export_ReverseNonequilibriumShearFlowGPU<ParticleLoader>(m,
                                                             "ReverseNonequilibriumShearFlowGPU");
    }
    } // namespace detail

    } // namespace md
    } // namespace hoomd
