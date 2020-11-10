// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "TwoStepRATTLENVEGPU.h"
#include "TwoStepRATTLENVEGPU.cuh"

namespace py = pybind11;
using namespace std;

/*! \file TwoStepRATTLENVEGPU.h
    \brief Contains code for the TwoStepRATTLENVEGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
*/
TwoStepRATTLENVEGPU::TwoStepRATTLENVEGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       std::shared_ptr<Manifold> manifold)
    : TwoStepRATTLENVE(sysdef, group, manifold), m_manifoldGPU( manifold->returnL(), manifold->returnR(), manifold->returnSurf() )
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepRATTLENVEGPU when CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepRATTLENVEGPU");
        }

    // initialize autotuner
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params.push_back(block_size);

    m_tuner_one.reset(new Autotuner(valid_params, 5, 100000, "rattle_nve_step_one", this->m_exec_conf));
    m_tuner_two.reset(new Autotuner(valid_params, 5, 100000, "rattle_nve_step_two", this->m_exec_conf));
    m_tuner_angular_one.reset(new Autotuner(valid_params, 5, 100000, "rattle_nve_angular_one", this->m_exec_conf));
    m_tuner_angular_two.reset(new Autotuner(valid_params, 5, 100000, "rattle_nve_angular_two", this->m_exec_conf));
    }
/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepRATTLENVEGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "NVE step 1");

    // access all the needed data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    gpu_rattle_nve_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_image.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     box,
                     m_deltaT,
                     m_limit,
                     m_limit_val,
                     m_tuner_one->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        gpu_rattle_nve_angular_step_one(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 m_group->getGPUPartition(),
                                 m_deltaT,
                                 1.0,
                                 m_tuner_angular_one->getParam());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();
        }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepRATTLENVEGPU::integrateStepTwo(unsigned int timestep)
    {
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "NVE step 2");

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    m_exec_conf->beginMultiGPU();
    m_tuner_two->begin();

    gpu_rattle_nve_step_two(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     d_net_force.data,
                     m_manifoldGPU,
                     m_eta,
                     m_deltaT,
                     m_limit,
                     m_limit_val,
                     m_zero_force,
                     m_tuner_two->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_two->end();
    m_exec_conf->endMultiGPU();

    if (m_aniso)
        {
        // second part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_two->begin();

        gpu_rattle_nve_angular_step_two(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 m_group->getGPUPartition(),
                                 m_deltaT,
                                 1.0,
                                 m_tuner_angular_two->getParam());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_two->end();
        m_exec_conf->endMultiGPU();
        }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void TwoStepRATTLENVEGPU::IncludeRATTLEForce(unsigned int timestep)
    {

    // access all the needed data
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>&  net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);

    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    unsigned int net_virial_pitch = net_virial.getPitch();

    // perform the update on the GPU
    m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    gpu_include_rattle_force_nve(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_net_force.data,
                     d_net_virial.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     net_virial_pitch,
                     m_manifoldGPU,
                     m_eta,
                     m_deltaT,
                     m_zero_force,
                     m_tuner_one->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    }

void export_TwoStepRATTLENVEGPU(py::module& m)
    {
    py::class_<TwoStepRATTLENVEGPU, TwoStepRATTLENVE, std::shared_ptr<TwoStepRATTLENVEGPU> >(m, "TwoStepRATTLENVEGPU")
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<Manifold> >())
        ;
    }
