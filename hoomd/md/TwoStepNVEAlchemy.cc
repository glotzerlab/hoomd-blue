// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNVEAlchemy.h"
#include "hoomd/VectorMath.h"

using namespace std;
namespace py = pybind11;

/*! \file TwoStepNVEAlchemy.h
    \brief Contains code for the TwoStepNVEAlchemy class
*/

namespace hoomd
    {

namespace md
    {

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param skip_restart Skip initialization of the restart information
*/
TwoStepNVEAlchemy::TwoStepNVEAlchemy(std::shared_ptr<SystemDefinition> sysdef,
                                     unsigned int alchemTimeFactor)
    : AlchemostatTwoStep(sysdef, alchemTimeFactor)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVEAlchemy" << endl;
    }

TwoStepNVEAlchemy::~TwoStepNVEAlchemy()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVEAlchemy" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.
*/
void TwoStepNVEAlchemy::integrateStepOne(uint64_t timestep)
    {
    if (timestep != m_nextAlchemTimeStep)
        return;

    m_nextAlchemTimeStep += m_nTimeFactor;

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
        // update velocity
        p += m_halfDeltaT * (netForce - mu - dUextdalpha);

        alpha->m_nextTimestep = m_nextAlchemTimeStep;
        }
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVEAlchemy::integrateStepTwo(uint64_t timestep)
    {
    if (timestep != (m_nextAlchemTimeStep - 1))
        return;

    Scalar dUextdalpha = Scalar(0);

    for (auto& alpha : m_alchemicalParticles)
        {
        Scalar& q = alpha->value;
        Scalar& p = alpha->momentum;

        const Scalar& invM = alpha->mass.y;
        const Scalar& mu = alpha->mu;
        const Scalar netForce = alpha->getNetForce(timestep + 1);

        // update velocity
        p += m_halfDeltaT * (netForce - mu - dUextdalpha);
        // update position
        q += m_halfDeltaT * p * invM;
        }
    }

namespace detail
    {

void export_TwoStepNVEAlchemy(py::module& m)
    {
    py::class_<TwoStepNVEAlchemy, AlchemostatTwoStep, std::shared_ptr<TwoStepNVEAlchemy>>(
        m,
        "TwoStepNVEAlchemy")
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int>());
    }

    } // end namespace detail

    } // end namespace md

    } // end namespace hoomd
