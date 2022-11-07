// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepBerendsen.h"

using namespace std;

/*! \file TwoStepBerendsen.cc
    \brief Definition of Berendsen thermostat
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to zero the velocities of
    \param group Group of particles on which this method will act
    \param thermo compute for thermodynamic quantities
    \param tau Berendsen time constant
    \param T Temperature set point
*/
TwoStepBerendsen::TwoStepBerendsen(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   std::shared_ptr<ComputeThermo> thermo,
                                   Scalar tau,
                                   std::shared_ptr<Variant> T)
    : TwoStepNVTBase(sysdef, group, thermo, T), m_tau(tau)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepBerendsen" << endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.berendsen: tau set less than 0.0" << endl;
    }

TwoStepBerendsen::~TwoStepBerendsen()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepBerendsen" << endl;
    }

std::array<Scalar, 2> TwoStepBerendsen::NVT_rescale_factor_one(uint64_t timestep)
    {
    m_thermo->compute(timestep);
    Scalar current_translation_T = m_thermo->getTranslationalTemperature();
    Scalar current_rotational_T = m_thermo->getRotationalTemperature();

    if((m_thermo->getTranslationalDOF() != 0 && m_thermo->getTranslationalKineticEnergy() == 0) ||
        (m_thermo->getRotationalDOF() != 0 && m_thermo->getRotationalKineticEnergy() == 0))
        {
        throw std::runtime_error("Bussi thermostat requires non-zero initial temperatures");
        }

    Scalar lambda_T = sqrt(Scalar(1.0) + m_deltaT / m_tau * ((*m_T)(timestep) / current_translation_T - Scalar(1.0)));
    Scalar lambda_R = sqrt(Scalar(1.0) + m_deltaT / m_tau * ((*m_T)(timestep) / current_rotational_T - Scalar(1.0)));

    return {lambda_T, lambda_R};
    }


namespace detail
    {
void export_Berendsen(pybind11::module& m)
    {
    pybind11::class_<TwoStepBerendsen, TwoStepNVTBase, std::shared_ptr<TwoStepBerendsen>>(
        m,
        "TwoStepBerendsen")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::shared_ptr<Variant>>())
        .def_property("tau", &TwoStepBerendsen::getTau, &TwoStepBerendsen::setTau);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
