// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveForceComputeGPU.h"
#include "ActiveForceComputeGPU.cuh"

#include <vector>
namespace py = pybind11;
using namespace std;

/*! \file ActiveForceComputeGPU.cc
    \brief Contains code for the ActiveForceComputeGPU class
*/

/*! \param seed required user-specified seed number for random number generator.
    \param f_list An array of (x,y,z) tuples for the active force vector for each individual particle.
    \param orientation_link if True then particle orientation is coupled to the active force vector. Only
    relevant for non-point-like anisotropic particles.
    /param orientation_reverse_link When True, the active force vector is coupled to particle orientation. Useful for
    for using a particle's orientation to log the active force vector.
    \param rotation_diff rotational diffusion constant for all particles.
    \param constraint specifies a constraint surface, to which particles are confined,
    such as update.constraint_ellipsoid.
*/
ActiveForceComputeGPU::ActiveForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<ParticleGroup> group,
                                        int seed,
                                        pybind11::list f_lst,
                                        bool orientation_link,
                                        bool orientation_reverse_link,
                                        Scalar rotation_diff,
                                        Scalar3 P,
                                        Scalar rx,
                                        Scalar ry,
                                        Scalar rz)
        : ActiveForceCompute(sysdef, group, seed, f_lst, orientation_link, orientation_reverse_link, rotation_diff, P, rx, ry, rz), m_block_size(256)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ActiveForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ActiveForceComputeGPU");
        }

    unsigned int N = m_pdata->getNGlobal();
    unsigned int group_size = m_group->getNumMembersGlobal();
    GPUArray<Scalar3> tmp_activeVec(N, m_exec_conf);
    GPUArray<Scalar> tmp_activeMag(N, m_exec_conf);
    GPUArray<unsigned int> tmp_groupTags(group_size, m_exec_conf);

        {
        ArrayHandle<Scalar3> old_activeVec(m_activeVec, access_location::host);
        ArrayHandle<Scalar> old_activeMag(m_activeMag, access_location::host);

        ArrayHandle<Scalar3> activeVec(tmp_activeVec, access_location::host);
        ArrayHandle<Scalar> activeMag(tmp_activeMag, access_location::host);
        ArrayHandle<unsigned int> groupTags(tmp_groupTags, access_location::host);

        // for each of the particles in the group
        for (unsigned int i = 0; i < group_size; i++)
            {
            unsigned int tag = m_group->getMemberTag(i);
            groupTags.data[i] = tag;
            activeMag.data[tag] = old_activeMag.data[i];
            activeVec.data[tag] = old_activeVec.data[i];
            }

        last_computed = 10;
        }

    m_activeVec.swap(tmp_activeVec);
    m_activeMag.swap(tmp_activeMag);
    m_groupTags.swap(tmp_groupTags);
    }

/*! This function sets appropriate active forces on all active particles.
*/
void ActiveForceComputeGPU::setForces()
    {
    //  array handles
    ArrayHandle<Scalar3> d_actVec(m_activeVec, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_actMag(m_activeMag, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_groupTags(m_groupTags, access_location::device, access_mode::read);

    // sanity check
    assert(d_force.data != NULL);
    assert(d_actVec.data != NULL);
    assert(d_actMag.data != NULL);
    assert(d_orientation.data != NULL);
    assert(d_rtag.data != NULL);
    assert(d_groupTags.data != NULL);
    bool orientationLink = (m_orientationLink == true);
    bool orientationReverseLink = (m_orientationReverseLink == true);
    unsigned int group_size = m_group->getNumMembers();
    unsigned int N = m_pdata->getN();

    gpu_compute_active_force_set_forces(group_size,
                                     d_rtag.data,
                                     d_groupTags.data,
                                     d_force.data,
                                     d_orientation.data,
                                     d_actVec.data,
                                     d_actMag.data,
                                     m_P,
                                     m_rx,
                                     m_ry,
                                     m_rz,
                                     orientationLink,
                                     orientationReverseLink,
                                     N,
                                     m_block_size);
    }

/*! This function applies rotational diffusion to all active particles
    \param timestep Current timestep
*/
void ActiveForceComputeGPU::rotationalDiffusion(unsigned int timestep)
    {
    //  array handles
    ArrayHandle<Scalar3> d_actVec(m_activeVec, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata -> getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_groupTags(m_groupTags, access_location::device, access_mode::read);

    assert(d_pos.data != NULL);

    bool is2D = (m_sysdef->getNDimensions() == 2);
    unsigned int group_size = m_group->getNumMembers();

    gpu_compute_active_force_rotational_diffusion(group_size,
                                                d_rtag.data,
                                                d_groupTags.data,
                                                d_pos.data,
                                                d_force.data,
                                                d_actVec.data,
                                                m_P,
                                                m_rx,
                                                m_ry,
                                                m_rz,
                                                is2D,
                                                m_rotationConst,
                                                timestep,
                                                m_seed,
                                                m_block_size);
    }

/*! This function sets an ellipsoid surface constraint for all active particles
*/
void ActiveForceComputeGPU::setConstraint()
    {
    EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);

    //  array handles
    ArrayHandle<Scalar3> d_actVec(m_activeVec, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata -> getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_groupTags(m_groupTags, access_location::device, access_mode::read);

    assert(d_pos.data != NULL);

    unsigned int group_size = m_group->getNumMembers();

    gpu_compute_active_force_set_constraints(group_size,
                                             d_rtag.data,
                                             d_groupTags.data,
                                             d_pos.data,
                                             d_force.data,
                                             d_actVec.data,
                                             m_P,
                                             m_rx,
                                             m_ry,
                                             m_rz,
                                             m_block_size);
    }

void export_ActiveForceComputeGPU(py::module& m)
    {
    py::class_< ActiveForceComputeGPU, std::shared_ptr<ActiveForceComputeGPU> >(m, "ActiveForceComputeGPU", py::base<ActiveForceCompute>())
        .def(py::init<  std::shared_ptr<SystemDefinition>,
                        std::shared_ptr<ParticleGroup>,
                        int,
                        pybind11::list,
                        bool,
                        bool,
                        Scalar,
                        Scalar3,
                        Scalar,
                        Scalar,
                        Scalar >())
    ;
    }
