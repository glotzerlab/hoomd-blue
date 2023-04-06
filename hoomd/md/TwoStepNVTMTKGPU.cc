// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNVTMTKGPU.h"
#include "TwoStepNPTMTKGPU.cuh"
#include "TwoStepNVEGPU.cuh"
#include "TwoStepNVTMTKGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

/*! \file TwoStepNVTMTKGPU.h
    \brief Contains code for the TwoStepNVTMTKGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
*/
TwoStepNVTMTKGPU::TwoStepNVTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   std::shared_ptr<ComputeThermo> thermo,
                                   Scalar tau,
                                   std::shared_ptr<Variant> T)
    : TwoStepNVTMTK(sysdef, group, thermo, tau, T)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create TwoStepNVTMTKGPU on a CPU device.");
        }

    // Initialize autotuners.
    m_tuner_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "nvt_mtk_step_one"));
    m_tuner_two.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "nvt_mtk_step_two"));
    m_tuner_angular_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "nvt_mtk_angular_one",
                                               5,
                                               true));
    m_tuner_angular_two.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "nvt_mtk_angular_two",
                                               5,
                                               true));
    m_autotuners.insert(m_autotuners.end(),
                        {m_tuner_one, m_tuner_two, m_tuner_angular_one, m_tuner_angular_two});
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   Nose-Hoover method
*/
void TwoStepNVTMTKGPU::integrateStepOne(uint64_t timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        throw std::runtime_error("Empty integration group.");
        }

    unsigned int group_size = m_group->getNumMembers();

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

        m_exec_conf->beginMultiGPU();

        // perform the update on the GPU
        m_tuner_one->begin();
        kernel::gpu_nvt_mtk_step_one(d_pos.data,
                                     d_vel.data,
                                     d_accel.data,
                                     d_image.data,
                                     d_index_array.data,
                                     group_size,
                                     box,
                                     m_tuner_one->getParam()[0],
                                     m_exp_thermo_fac,
                                     m_deltaT,
                                     m_group->getGPUPartition());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_one->end();

        m_exec_conf->endMultiGPU();
        }

    if (m_aniso)
        {
        // angular degrees of freedom, step one
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
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        const Scalar exp_fac = exp(-m_deltaT / Scalar(2.0) * m_thermostat.xi_rot);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();
        kernel::gpu_nve_angular_step_one(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         exp_fac,
                                         m_tuner_angular_one->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();
        }

    // advance thermostat
    advanceThermostat(timestep, false);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNVTMTKGPU::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();

        // perform the update on the GPU
        m_tuner_two->begin();
        kernel::gpu_nvt_mtk_step_two(d_vel.data,
                                     d_accel.data,
                                     d_index_array.data,
                                     group_size,
                                     d_net_force.data,
                                     m_tuner_two->getParam()[0],
                                     m_deltaT,
                                     m_exp_thermo_fac,
                                     m_group->getGPUPartition());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_two->end();

        m_exec_conf->endMultiGPU();
        }

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

        Scalar exp_fac = exp(-m_deltaT / Scalar(2.0) * m_thermostat.xi_rot);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_two->begin();
        kernel::gpu_nve_angular_step_two(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         exp_fac,
                                         m_tuner_angular_two->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_angular_two->end();
        m_exec_conf->endMultiGPU();
        }
    }

namespace detail
    {
void export_TwoStepNVTMTKGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepNVTMTKGPU, TwoStepNVTMTK, std::shared_ptr<TwoStepNVTMTKGPU>>(
        m,
        "TwoStepNVTMTKGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::shared_ptr<Variant>>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
