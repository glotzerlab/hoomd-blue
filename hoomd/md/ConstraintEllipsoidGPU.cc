// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstraintEllipsoidGPU.h"
#include "ConstraintEllipsoidGPU.cuh"

namespace py = pybind11;

using namespace std;

/*! \file ConstraintEllipsoidGPU.cc
    \brief Contains code for the ConstraintEllipsoidGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the Ellipsoid
    \param rx radius of the Ellipsoid in the X direction
    \param ry radius of the Ellipsoid in the Y direction
    \param rz radius of the Ellipsoid in the Z direction
    NOTE: For the algorithm to work, we must have _rx >= _rz, ry >= _rz, and _rz > 0.
*/
ConstraintEllipsoidGPU::ConstraintEllipsoidGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   Scalar3 P,
                                   Scalar rx,
                                   Scalar ry,
                                   Scalar rz)
        : ConstraintEllipsoid(sysdef, group, P, rx, ry, rz), m_block_size(256)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ConstraintEllipsoidGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ConstraintEllipsoidGPU");
        }
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintEllipsoidGPU::update(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("ConstraintEllipsoid");

    assert(m_pdata);

    // access the particle data arrays
    const GlobalArray< unsigned int >& group_members = m_group->getIndexArray();
    ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

    // run the kernel in parallel on all GPUs
    gpu_compute_constraint_ellipsoid_constraint(d_group_members.data,
                                         m_group->getNumMembers(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_P,
                                         m_rx,
                                         m_ry,
                                         m_rz,
                                         m_block_size);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_ConstraintEllipsoidGPU(py::module& m)
    {
    py::class_< ConstraintEllipsoidGPU, std::shared_ptr<ConstraintEllipsoidGPU> >(m, "ConstraintEllipsoidGPU", py::base<ConstraintEllipsoid>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                             std::shared_ptr<ParticleGroup>,
                             Scalar3,
                             Scalar,
                             Scalar,
                             Scalar >())
                             ;
    }
