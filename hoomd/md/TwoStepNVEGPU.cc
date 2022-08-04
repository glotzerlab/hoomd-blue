// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNVEGPU.h"
#include "TwoStepNVEGPU.cuh"

using namespace std;

/*! \file TwoStepNVEGPU.h
    \brief Contains code for the TwoStepNVEGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
*/
TwoStepNVEGPU::TwoStepNVEGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group)
    : TwoStepNVE(sysdef, group)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot initialize TwoStepNVEGPU on a CPU device.");
        }

    // Initialize autotuners.
    m_tuner_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "nve_step_one"));
    m_tuner_two.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "nve_step_two"));
    m_tuner_angular_one.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "nve_angular_one"));
    m_tuner_angular_two.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "nve_angular_two"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_one, m_tuner_two, m_tuner_angular_one, m_tuner_angular_two});
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.
*/
void TwoStepNVEGPU::integrateStepOne(uint64_t timestep)
    {
    // access all the needed data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    BoxDim box = m_pdata->getBox();
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // perform the update on the GPU
    m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    kernel::gpu_nve_step_one(d_pos.data,
                             d_vel.data,
                             d_accel.data,
                             d_image.data,
                             d_index_array.data,
                             m_group->getGPUPartition(),
                             box,
                             m_deltaT,
                             m_limit,
                             m_limit_val,
                             m_zero_force,
                             m_tuner_one->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        kernel::gpu_nve_angular_step_one(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         1.0,
                                         m_tuner_angular_one->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();
        }
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNVEGPU::integrateStepTwo(uint64_t timestep)
    {
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // perform the update on the GPU
    m_exec_conf->beginMultiGPU();
    m_tuner_two->begin();

    kernel::gpu_nve_step_two(d_vel.data,
                             d_accel.data,
                             d_index_array.data,
                             m_group->getGPUPartition(),
                             d_net_force.data,
                             m_deltaT,
                             m_limit,
                             m_limit_val,
                             m_zero_force,
                             m_tuner_two->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_two->end();
    m_exec_conf->endMultiGPU();

    if (m_aniso)
        {
        // second part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_two->begin();

        kernel::gpu_nve_angular_step_two(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         1.0,
                                         m_tuner_angular_two->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_two->end();
        m_exec_conf->endMultiGPU();
        }
    }

namespace detail
    {
void export_TwoStepNVEGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepNVEGPU, TwoStepNVE, std::shared_ptr<TwoStepNVEGPU>>(m, "TwoStepNVEGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
