// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstTorqueCompute.h"

namespace py = pybind11;

using namespace std;

/*! \file ConstTorqueCompute.cc
    \brief Contains code for the ConstTorqueCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstTorqueCompute::ConstTorqueCompute(std::shared_ptr<SystemDefinition> sysdef, Scalar fx, Scalar fy, Scalar fz)
        : ForceCompute(sysdef), m_fx(fx), m_fy(fy), m_fz(fz)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstTorqueCompute" << endl;

    setTorque(fx,fy,fz);
    }

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group A group of particles
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstTorqueCompute::ConstTorqueCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz)
        : ForceCompute(sysdef), m_fx(fx), m_fy(fy), m_fz(fz)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstTorqueCompute" << endl;

    setGroupTorque(group,fx,fy,fz);
    }

ConstTorqueCompute::~ConstTorqueCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstTorqueCompute" << endl;
    }

/*! \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
*/
void ConstTorqueCompute::setTorque(Scalar fx, Scalar fy, Scalar fz)
    {
    assert(m_pdata != NULL);

    m_fx = fx;
    m_fy = fy;
    m_fz = fz;

    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);
    //Don't need to zero data for force calculation.

    assert(h_torque.data);

    // setting the force is simple, just fill out every element of the force array
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_torque.data[i].x = fx;
        h_torque.data[i].y = fy;
        h_torque.data[i].z = fz;
        h_torque.data[i].w = 0;
        }
   }

/*! \param i Index of the particle to set
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
*/
void ConstTorqueCompute::setParticleTorque(unsigned int i, Scalar fx, Scalar fy, Scalar fz)
    {

    assert(m_pdata != NULL);
    assert(i < m_pdata->getN());

    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);
    assert(h_torque.data);

    h_torque.data[i].x = fx;
    h_torque.data[i].y = fy;
    h_torque.data[i].z = fz;
    h_torque.data[i].w = 0;
    }

/*! \param group Group to set the force for
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
*/
void ConstTorqueCompute::setGroupTorque(std::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz)
    {
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);

    m_fx = fx;
    m_fy = fy;
    m_fz = fz;
    m_group = group;

    // Reset force array
    for (unsigned int i = 0;i < m_pdata->getN();i++)
        {
        h_torque.data[i].x = 0;
        h_torque.data[i].y = 0;
        h_torque.data[i].z = 0;
        h_torque.data[i].w = 0;
        }

    for (unsigned int i = 0; i < group->getNumMembers(); i++)
        {
        // get the index for the current group member
        unsigned int idx = group->getMemberIndex(i);
        h_torque.data[idx].x = fx;
        h_torque.data[idx].y = fy;
        h_torque.data[idx].z = fz;
        h_torque.data[idx].w = 0;
        }

    }

void ConstTorqueCompute::rearrangeTorques()
    {
    if (m_group)
        // set force only on group of particles
        setGroupTorque(m_group, m_fx, m_fy, m_fz);
    else
        // set force on all particles
        setTorque(m_fx, m_fy, m_fz);
    }

/*! This function calls rearrangeTorques() whenever the particles have been sorted
    \param timestep Current timestep
*/
void ConstTorqueCompute::computeTorques(unsigned int timestep)
    {
    if (m_particles_sorted==true) rearrangeTorques();
    }


void export_ConstTorqueCompute(py::module& m)
    {
    py::class_< ConstTorqueCompute, std::shared_ptr<ConstTorqueCompute> >(m,"ConstTorqueCompute",py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, Scalar, Scalar >())
    .def("setTorque", &ConstTorqueCompute::setTorque)
    .def("setGroupTorque", &ConstTorqueCompute::setGroupTorque)
    ;
    }
