// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser



#include "TwoStepNVTMTKGPU.h"
#include "TwoStepNVTMTKGPU.cuh"
#include "TwoStepNVEGPU.cuh"
#include "TwoStepNPTMTKGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
using namespace std;

/*! \file TwoStepNVTMTKGPU.h
    \brief Contains code for the TwoStepNVTMTKGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepNVTMTKGPU::TwoStepNVTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ComputeThermo> thermo,
                             Scalar tau,
                             std::shared_ptr<Variant> T,
                             const std::string& suffix)
    : TwoStepNVTMTK(sysdef, group, thermo, tau, T, suffix)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepNVTMTKPU when CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepNVTMTKGPU");
        }

    // initialize autotuner
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params.push_back(block_size);

    m_tuner_one.reset(new Autotuner(valid_params, 5, 100000, "nvt_mtk_step_one", this->m_exec_conf));
    m_tuner_two.reset(new Autotuner(valid_params, 5, 100000, "nvt_mtk_step_two", this->m_exec_conf));
    m_tuner_angular_one.reset(new Autotuner(valid_params, 5, 100000, "nvt_mtk_angular_one", this->m_exec_conf));
    m_tuner_angular_two.reset(new Autotuner(valid_params, 5, 100000, "nvt_mtk_angular_two", this->m_exec_conf));
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover method
*/
void TwoStepNVTMTKGPU::integrateStepOne(unsigned int timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        m_exec_conf->msg->error() << "integrate.nvt(): Integration group empty." << std::endl;
        throw std::runtime_error("Error during NVT integration.");
        }

    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        {
        m_prof->push(m_exec_conf, "NVT MTK step 1");
        }

        {
        // access all the needed data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

        BoxDim box = m_pdata->getBox();
        ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();

        // perform the update on the GPU
        m_tuner_one->begin();
        gpu_nvt_mtk_step_one(d_pos.data,
                         d_vel.data,
                         d_accel.data,
                         d_image.data,
                         d_index_array.data,
                         group_size,
                         box,
                         m_tuner_one->getParam(),
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
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);
        ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        IntegratorVariables v = getIntegratorVariables();
        Scalar xi_rot = v.variable[2];
        Scalar exp_fac = exp(-m_deltaT/Scalar(2.0)*xi_rot);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();
        gpu_nve_angular_step_one(d_orientation.data,
                             d_angmom.data,
                             d_inertia.data,
                             d_net_torque.data,
                             d_index_array.data,
                             m_group->getGPUPartition(),
                             m_deltaT,
                             exp_fac,
                             m_tuner_angular_one->getParam());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();
        }

    // advance thermostat
    advanceThermostat(timestep, false);

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNVTMTKGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "NVT MTK step 2");

    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();

        // perform the update on the GPU
        m_tuner_two->begin();
        gpu_nvt_mtk_step_two(d_vel.data,
                         d_accel.data,
                         d_index_array.data,
                         group_size,
                         d_net_force.data,
                         m_tuner_two->getParam(),
                         m_deltaT,
                         m_exp_thermo_fac,
                         m_group->getGPUPartition());

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_two->end();

        m_exec_conf->endMultiGPU();
        }

    if (m_aniso)
        {
        // second part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        IntegratorVariables v = getIntegratorVariables();
        Scalar xi_rot = v.variable[2];
        Scalar exp_fac = exp(-m_deltaT/Scalar(2.0)*xi_rot);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_two->begin();
        gpu_nve_angular_step_two(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 m_group->getGPUPartition(),
                                 m_deltaT,
                                 exp_fac,
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

void export_TwoStepNVTMTKGPU(py::module& m)
    {
    py::class_<TwoStepNVTMTKGPU, std::shared_ptr<TwoStepNVTMTKGPU> >(m, "TwoStepNVTMTKGPU", py:: base<TwoStepNVTMTK>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                          std::shared_ptr<ParticleGroup>,
                          std::shared_ptr<ComputeThermo>,
                          Scalar,
                          std::shared_ptr<Variant>,
                          const std::string&
                          >())
        ;
    }
