// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepBerendsen.h"
#ifdef ENABLE_CUDA
#include "TwoStepBerendsenGPU.cuh"
#endif

using namespace std;
namespace py = pybind11;

/*! \file TwoStepBerendsen.cc
    \brief Definition of Berendsen thermostat
*/

// ********************************
// here follows the code for Berendsen on the CPU

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
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_tau(tau), m_T(T),
      m_warned_aniso(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepBerendsen" << endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.berendsen: tau set less than 0.0" << endl;
    }

TwoStepBerendsen::~TwoStepBerendsen()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepBerendsen" << endl;
    }

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TwoStepBerendsen::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (m_aniso && !m_warned_aniso)
        {
        m_exec_conf->msg->warning() << "integrate.berendsen: this integrator "
            "does not support anisotropic degrees of freedom" << endl;
        m_warned_aniso = true;
        }

    // profile this step
    if (m_prof)
        m_prof->push("Berendsen step 1");

    // compute the current thermodynamic properties and get the temperature
    m_thermo->compute(timestep);
    Scalar curr_T = m_thermo->getTranslationalTemperature();

    // compute the value of lambda for the current timestep
    Scalar lambda = sqrt(Scalar(1.0) + m_deltaT / m_tau * (m_T->getValue(timestep) / curr_T - Scalar(1.0)));

    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);


    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // advance velocity forward by half a timestep and position forward by a full timestep
        h_vel.data[j].x = lambda * (h_vel.data[j].x + h_accel.data[j].x * m_deltaT * Scalar(1.0 / 2.0));
        h_pos.data[j].x += h_vel.data[j].x * m_deltaT;

        h_vel.data[j].y = lambda * (h_vel.data[j].y + h_accel.data[j].y * m_deltaT * Scalar(1.0 / 2.0));
        h_pos.data[j].y += h_vel.data[j].y * m_deltaT;

        h_vel.data[j].z = lambda * (h_vel.data[j].z + h_accel.data[j].z * m_deltaT * Scalar(1.0 / 2.0));
        h_pos.data[j].z += h_vel.data[j].z * m_deltaT;
        }

    /* particles may have been moved slightly outside the box by the above steps so we should wrap
        them back into place */
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        box.wrap(h_pos.data[j], h_image.data[j]);
        }

    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current timestep
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepBerendsen::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    // access the force data
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle< Scalar4 > h_net_force(net_force, access_location::host, access_mode::read);

    // profile this step
    if (m_prof)
        m_prof->push("Berendsen step 2");

    // integrate the particle velocities to timestep+1
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // calculate the acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = h_net_force.data[j].x * minv;
        h_accel.data[j].y = h_net_force.data[j].y * minv;
        h_accel.data[j].z = h_net_force.data[j].z * minv;

        // update the velocity
        h_vel.data[j].x += h_accel.data[j].x * m_deltaT / Scalar(2.0);
        h_vel.data[j].y += h_accel.data[j].y * m_deltaT / Scalar(2.0);
        h_vel.data[j].z += h_accel.data[j].z * m_deltaT / Scalar(2.0);
        }

    }

void export_Berendsen(py::module& m)
    {
    py::class_<TwoStepBerendsen, std::shared_ptr<TwoStepBerendsen> >(m, "TwoStepBerendsen", py::base<IntegrationMethodTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr<ParticleGroup>,
                         std::shared_ptr<ComputeThermo>,
                         Scalar,
                         std::shared_ptr<Variant>
                         >())
        .def("setT", &TwoStepBerendsen::setT)
        .def("setTau", &TwoStepBerendsen::setTau)
        ;
    }
