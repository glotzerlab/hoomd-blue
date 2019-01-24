// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "OneDConstraintGPU.h"
#include "OneDConstraintGPU.cuh"

namespace py = pybind11;

using namespace std;

/*! \file OneDConstraintGPU.cc
    \brief Contains code for the OneDConstraintGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the sphere
    \param r radius of the sphere
*/
OneDConstraintGPU::OneDConstraintGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<ParticleGroup> group,
                                         Scalar3 constraint_vec)
        : OneDConstraint(sysdef, group, constraint_vec), m_block_size(256)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a OneDConstraintGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing OneDConstraintGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "oneD_constraint", this->m_exec_conf));

    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void OneDConstraintGPU::computeForces(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push(m_exec_conf, "OneDConstraint");

    assert(m_pdata);

    // access the particle data arrays
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

    const GlobalArray< unsigned int >& group_members = m_group->getIndexArray();
    ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    // run the kernel in parallel on all GPUs
    m_tuner->begin();
    gpu_compute_one_d_constraint_forces(d_force.data,
                                         d_virial.data,
                                         m_virial.getPitch(),
                                         d_group_members.data,
                                         m_group->getNumMembers(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         d_vel.data,
                                         d_net_force.data,
                                         m_deltaT,
                                         m_tuner->getParam(),
                                         m_vec);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_OneDConstraintGPU(py::module& m)
    {
    py::class_< OneDConstraintGPU, std::shared_ptr<OneDConstraintGPU> >(m, "OneDConstraintGPU", py::base<ForceConstraint>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                   std::shared_ptr<ParticleGroup>,
                   Scalar3 >())
    ;
    }
