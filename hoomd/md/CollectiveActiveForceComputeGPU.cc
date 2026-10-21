// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CollectiveActiveForceComputeGPU.h"
#include "CollectiveActiveForceComputeGPU.cuh"

/*! \file CollectiveActiveForceComputeGPU.cc
    \brief Contains code for the CollectiveActiveForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef The system definition
    \param group Particle group the force is applied to
*/
CollectiveActiveForceComputeGPU::CollectiveActiveForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group)
    : CollectiveActiveForceCompute(sysdef, group)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a CollectiveActiveForceComputeGPU with no GPU in the execution "
               "configuration"
            << std::endl;
        throw std::runtime_error("Error initializing CollectiveActiveForceComputeGPU");
        }

    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "collective_active_force"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_force});
    }

/*! Applies the active force vector of each particle type, unchanged and in the box frame, to
    every member of the group. Particle orientations are not read, and no torque is applied.
 */
void CollectiveActiveForceComputeGPU::setForces()
    {
    // getNumMembers might allocate the index array. Access it first, then aquire the handles.
    const unsigned int group_size = m_group->getNumMembers();

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    m_force.zeroFill();
    // this force never applies a torque
    m_torque.zeroFill();

    ArrayHandle<Scalar4> d_f_actVec(m_f_activeVec, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    assert(d_force.data != NULL);
    assert(d_f_actVec.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_index_array.data != NULL);

    m_tuner_force->begin();

    kernel::gpu_compute_collective_active_force_set_forces(group_size,
                                                           d_index_array.data,
                                                           d_force.data,
                                                           d_pos.data,
                                                           d_f_actVec.data,
                                                           m_tuner_force->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_force->end();
    }

namespace detail
    {
void export_CollectiveActiveForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<CollectiveActiveForceComputeGPU,
                     CollectiveActiveForceCompute,
                     std::shared_ptr<CollectiveActiveForceComputeGPU>>(
        m,
        "CollectiveActiveForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
