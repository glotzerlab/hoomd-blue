// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstraintSphereGPU.h"
#include "ConstraintSphereGPU.cuh"

#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace boost::python;
using namespace boost;

using namespace std;

/*! \file ConstraintSphereGPU.cc
    \brief Contains code for the ConstraintSphereGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the sphere
    \param r radius of the sphere
*/
ConstraintSphereGPU::ConstraintSphereGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                         boost::shared_ptr<ParticleGroup> group,
                                         Scalar3 P,
                                         Scalar r)
        : ConstraintSphere(sysdef, group, P, r), m_block_size(256)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ConstraintSphereGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ConstraintSphereGPU");
        }
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintSphereGPU::computeForces(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push(m_exec_conf, "ConstraintSphere");

    assert(m_pdata);

    // access the particle data arrays
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

    const GPUArray< unsigned int >& group_members = m_group->getIndexArray();
    ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    // run the kernel in parallel on all GPUs
    gpu_compute_constraint_sphere_forces(d_force.data,
                                         d_virial.data,
                                         m_virial.getPitch(),
                                         d_group_members.data,
                                         m_group->getNumMembers(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         d_vel.data,
                                         d_net_force.data,
                                         m_P,
                                         m_r,
                                         m_deltaT,
                                         m_block_size);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_ConstraintSphereGPU()
    {
    class_< ConstraintSphereGPU, boost::shared_ptr<ConstraintSphereGPU>, bases<ConstraintSphere>, boost::noncopyable >
    ("ConstraintSphereGPU", init< boost::shared_ptr<SystemDefinition>,
                                  boost::shared_ptr<ParticleGroup>,
                                  Scalar3,
                                  Scalar >())
    ;
    }
