// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNVTAlchemy.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/AlchemostatTwoStep.h"

namespace py = pybind11;

/*! \file TwoStepNVTAlchemy.cc
    \brief Contains code for the TwoStepNVTAlchemy class
*/

namespace hoomd
    {

namespace md
    {

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param skip_restart Skip initialization of the restart information
*/
TwoStepNVTAlchemy::TwoStepNVTAlchemy(std::shared_ptr<SystemDefinition> sysdef,
                                     unsigned int alchemTimeFactor,
                                     std::shared_ptr<Variant> T)
    : AlchemostatTwoStep(sysdef, alchemTimeFactor), m_Q(1.0), m_alchem_KE(0.0), m_T(T)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTAlchemy" << std::endl;

    m_thermostat.xi = 0;
    m_thermostat.eta = 0;

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        // This code is not tested or validated with MPI.
        throw std::runtime_error("Alchemical NVT integration method does not support MPI.");
        }
#endif
    }

TwoStepNVTAlchemy::~TwoStepNVTAlchemy()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTAlchemy" << std::endl;
    }

void TwoStepNVTAlchemy::integrateStepOne(uint64_t timestep)
    {
    if (timestep != m_nextAlchemTimeStep)
        return;

    m_exec_conf->msg->notice(10) << "TwoStepNVTAlchemy: 1st Alchemcial Half Step" << std::endl;

    m_nextAlchemTimeStep += m_nTimeFactor;

    m_alchem_KE = Scalar(0);

    Scalar dUextdalpha = Scalar(0);

    for (auto& alpha : m_alchemicalParticles)
        {
        Scalar& q = alpha->value;
        Scalar& p = alpha->momentum;

        const Scalar& invM = alpha->mass.y;
        const Scalar& mu = alpha->mu;
        const Scalar netForce = alpha->getNetForce(timestep);

        // update position
        q += m_halfDeltaT * p * invM;
        // update momentum
        p += m_halfDeltaT * (netForce - mu - dUextdalpha);
        // rescale velocity
        p *= exp(-m_halfDeltaT * m_thermostat.xi);
        m_alchem_KE += Scalar(0.5) * p * p * invM;

        alpha->m_nextTimestep = m_nextAlchemTimeStep;
        }

    advanceThermostat(timestep);
    }

void TwoStepNVTAlchemy::integrateStepTwo(uint64_t timestep)
    {
    if (timestep != (m_nextAlchemTimeStep - 1))
        return;

    m_exec_conf->msg->notice(10) << "TwoStepNVTAlchemy: 2nd Alchemcial Half Step" << std::endl;

    m_alchem_KE = Scalar(0);

    Scalar dUextdalpha = Scalar(0);

    for (auto& alpha : m_alchemicalParticles)
        {
        Scalar& q = alpha->value;
        Scalar& p = alpha->momentum;

        const Scalar& invM = alpha->mass.y;
        const Scalar& mu = alpha->mu;
        const Scalar netForce = alpha->getNetForce(timestep + 1);

        // rescale velocity
        p *= exp(-m_halfDeltaT * m_thermostat.xi);
        // update momentum
        p += m_halfDeltaT * (netForce - mu - dUextdalpha);
        // update position
        q += m_halfDeltaT * p * invM;
        m_alchem_KE += Scalar(0.5) * p * p * invM;
        }
    }

void TwoStepNVTAlchemy::advanceThermostat(uint64_t timestep, bool broadcast)
    {
    // update the state variables Xi and eta
    Scalar half_delta_xi = m_halfDeltaT
                           * ((Scalar(2) * m_alchem_KE)
                              - (Scalar(m_alchemicalParticles.size()) * m_T->operator()(timestep)))
                           / m_Q;
    m_thermostat.eta += (half_delta_xi + m_thermostat.xi) * m_deltaT * m_nTimeFactor;
    m_thermostat.xi += half_delta_xi + half_delta_xi;
    }

namespace detail
    {

void export_TwoStepNVTAlchemy(py::module& m)
    {
    py::class_<TwoStepNVTAlchemy, AlchemostatTwoStep, std::shared_ptr<TwoStepNVTAlchemy>>(
        m,
        "TwoStepNVTAlchemy")
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int, std::shared_ptr<Variant>>())
        .def_property("alchemical_kT", &TwoStepNVTAlchemy::getT, &TwoStepNVTAlchemy::setT)
        .def_property("Q", &TwoStepNVTAlchemy::getQ, &TwoStepNVTAlchemy::setQ);
    }

    } // end namespace detail

    } // end namespace md

    } // end namespace hoomd
