// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstForceCompute.h"

namespace py = pybind11;

using namespace std;

/*! \file ConstForceCompute.cc
    \brief Contains code for the ConstForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstForceCompute::ConstForceCompute(std::shared_ptr<SystemDefinition> sysdef,
        Scalar fx, Scalar fy, Scalar fz,
        Scalar tx, Scalar ty, Scalar tz)
        : ForceCompute(sysdef), m_fx(fx), m_fy(fy), m_fz(fz), m_tx(tx), m_ty(ty), m_tz(tz), m_need_rearrange_forces(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstForceCompute" << endl;


    setForce(fx,fy,fz,tx,ty,tz);
    }

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group A group of particles
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstForceCompute::ConstForceCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group,
        Scalar fx, Scalar fy, Scalar fz,
        Scalar tx, Scalar ty, Scalar tz)
        : ForceCompute(sysdef), m_fx(fx), m_fy(fy), m_fz(fz), m_tx(tx), m_ty(ty), m_tz(tz)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstForceCompute" << endl;

    setGroupForce(group,fx,fy,fz,tz,ty,tz);
    }

ConstForceCompute::~ConstForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstForceCompute" << endl;
    }

/*! \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \param tx x-component of the torque vector
    \param ty y-component of the torque vector
    \param tz z-component of the torque vector
*/
void ConstForceCompute::setForce(Scalar fx, Scalar fy, Scalar fz, Scalar tx, Scalar ty, Scalar tz)
    {
    assert(m_pdata != NULL);

    m_fx = fx;
    m_fy = fy;
    m_fz = fz;

    m_tx = tx;
    m_ty = ty;
    m_tz = tz;

    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);
    //Don't need to zero data for force calculation.

    assert(h_force.data);
    assert(h_torque.data);

    // setting the force and torque is simple, just fill out every element of the force array
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_force.data[i].x = fx;
        h_force.data[i].y = fy;
        h_force.data[i].z = fz;
        h_force.data[i].w = 0;

        h_torque.data[i].x = tx;
        h_torque.data[i].y = ty;
        h_torque.data[i].z = tz;
        h_torque.data[i].w = 0;

        }
   }

/*! \param tag Tag of the particle to set
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \param tx x-component of the torque vector
    \param ty y-component of the torque vector
    \param tz z-component of the torque vector
*/
void ConstForceCompute::setParticleForce(unsigned int tag, Scalar fx, Scalar fy, Scalar fz, Scalar tx, Scalar ty, Scalar tz)
    {
    if (tag > m_pdata->getMaximumTag())
        {
        m_exec_conf->msg->error() << "Particle tag needs to be smaller or equal the maximum tag in the system." << std::endl;
        throw std::runtime_error("Error setting particle forces.\n");
        }

    m_forces[tag] = vec3<Scalar>(fx,fy,fz);
    m_torques[tag] = vec3<Scalar>(tx,ty,tz);

    m_need_rearrange_forces = true;
    }

/*! \param group Group to set the force or torque for
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \param tx x-component of the torque vector
    \param ty y-component of the torque vector
    \param tz z-component of the torque vector
*/
void ConstForceCompute::setGroupForce(std::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz, Scalar tx, Scalar ty, Scalar tz)
    {
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);

    m_fx = fx;
    m_fy = fy;
    m_fz = fz;
    m_tx = tx;
    m_ty = ty;
    m_tz = tz;
    m_group = group;

    // Reset force and torque arrays
    for (unsigned int i = 0;i < m_pdata->getN();i++)
        {
        h_force.data[i].x = 0;
        h_force.data[i].y = 0;
        h_force.data[i].z = 0;
        h_force.data[i].w = 0;

        h_torque.data[i].x = 0;
        h_torque.data[i].y = 0;
        h_torque.data[i].z = 0;
        h_torque.data[i].w = 0;

        }

    for (unsigned int i = 0; i < group->getNumMembers(); i++)
        {
        // get the index for the current group member
        unsigned int idx = group->getMemberIndex(i);
        h_force.data[idx].x = fx;
        h_force.data[idx].y = fy;
        h_force.data[idx].z = fz;
        h_force.data[idx].w = 0;

        h_torque.data[idx].x = tx;
        h_torque.data[idx].y = ty;
        h_torque.data[idx].z = tz;
        h_torque.data[idx].w = 0;

        }

    }

void ConstForceCompute::rearrangeForces()
    {
    if (m_group)
        {
        // set force only on group of particles
        setGroupForce(m_group, m_fx, m_fy, m_fz, m_tx, m_ty, m_tz);
        }
    else
        {
        // set force on all particles
        setForce(m_fx, m_fy, m_fz, m_tx, m_ty, m_tz);
        }

    if (m_forces.size())
        {
        assert(m_forces.size() == m_torques.size());

        ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::readwrite);
        ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::readwrite);

        assert(h_force.data);
        assert(h_torque.data);

        for (auto it = m_forces.begin(); it != m_forces.end(); ++it)
            {
            unsigned int tag = it->first;
            unsigned int idx = m_pdata->getRTag(tag);
            bool ptl_local = (idx < m_pdata->getN());
            vec3<Scalar> f = it->second;
            vec3<Scalar> t = m_torques[it->first];

            if (ptl_local)
                {
                h_force.data[idx].x = f.x;
                h_force.data[idx].y = f.y;
                h_force.data[idx].z = f.z;
                h_force.data[idx].w = 0;

                h_torque.data[idx].x = t.x;
                h_torque.data[idx].y = t.y;
                h_torque.data[idx].z = t.z;
                h_torque.data[idx].w = 0;
                }
            }
        }

    m_need_rearrange_forces = false;
    }

/*! This function calls rearrangeForces() whenever the particles have been sorted
    \param timestep Current timestep
*/
void ConstForceCompute::computeForces(unsigned int timestep)
    {
    if (m_particles_sorted || m_need_rearrange_forces)
        rearrangeForces();

    // execute python callback to update the forces, if present
    if (m_callback && m_callback != py::none())
        {
        m_callback(timestep);
        }
    }


void export_ConstForceCompute(py::module& m)
    {
    py::class_< ConstForceCompute, std::shared_ptr<ConstForceCompute> >(m,"ConstForceCompute",py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar >())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar >())
    .def("setForce", &ConstForceCompute::setForce)
    .def("setGroupForce", &ConstForceCompute::setGroupForce)
    .def("setParticleForce", &ConstForceCompute::setParticleForce)
    .def("setCallback", &ConstForceCompute::setCallback)
    ;
    }
