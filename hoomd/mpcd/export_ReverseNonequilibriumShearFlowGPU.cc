// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ParticleLoader.h"
#include "hoomd/md/ReverseNonequilibriumShearFlowGPU.h"

namespace hoomd
    {
template class md::ReverseNonequilibriumShearFlowGPU<mpcd::ParticleLoader>;
namespace mpcd
    {
namespace detail
    {
void export_ReverseNonequilibriumShearFlowGPU(pybind11::module& m)
    {
    md::detail::export_ReverseNonequilibriumShearFlowGPU<ParticleLoader>(
        m,
        "ReverseNonequilibriumShearFlowGPU");
    }
    } // namespace detail

    } // namespace mpcd
    } // namespace hoomd
