// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepBerendsenGPU.cc
    \brief Defines TwoStepBerendsenGPU
*/

#include "TwoStepBerendsenGPU.h"

#include <functional>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to which the Berendsen thermostat will be applied
    \param group Group of particles to which the Berendsen thermostat will be applied
    \param thermo Compute for thermodynamic properties
    \param tau Time constant for Berendsen thermostat
    \param T Set temperature
*/
TwoStepBerendsenGPU::TwoStepBerendsenGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<ParticleGroup> group,
                                         std::shared_ptr<ComputeThermo> thermo,
                                         Scalar tau,
                                         std::shared_ptr<Variant> T)
    :  TwoStepNVTBase(sysdef, group, thermo, T), TwoStepBerendsen(sysdef, group, thermo, tau, T), TwoStepNVTBaseGPU(sysdef, group, thermo, T)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create BerendsenGPU on a CPU device.");
        }
    }


namespace detail
    {
void export_BerendsenGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepBerendsenGPU, TwoStepBerendsen, TwoStepNVTBaseGPU, std::shared_ptr<TwoStepBerendsenGPU>>(
        m,
        "TwoStepBerendsenGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::shared_ptr<Variant>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
