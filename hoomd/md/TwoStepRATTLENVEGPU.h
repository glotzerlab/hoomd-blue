// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepRATTLENVE.h"

#ifndef __TWO_STEP_RATTLE_NVE_GPU_H__
#define __TWO_STEP_RATTLE_NVE_GPU_H__

/*! \file TwoStepRATTLENVEGPU.h
    \brief Declares the TwoStepRATTLENVEGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#include "hoomd/Autotuner.h"

namespace py = pybind11;
using namespace std;

//! Integrates part of the system forward in two steps in the NVE ensemble on the GPU
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface, runs on the GPU

    \ingroup updaters
*/
template<class Manifold>
class PYBIND11_EXPORT TwoStepRATTLENVEGPU : public TwoStepRATTLENVE<Manifold>
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepRATTLENVEGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, Manifold manifold)
        : TwoStepRATTLENVE<Manifold>(sysdef, group, manifold)
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

        virtual ~TwoStepRATTLENVEGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Includes the RATTLE forces to the virial/net force
        virtual void IncludeRATTLEForce(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepRATTLENVE<Manifold>::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_two->setPeriod(period);
            m_tuner_two->setEnabled(enable);
            m_tuner_angular_one->setPeriod(period);
            m_tuner_angular_one->setEnabled(enable);
            m_tuner_angular_two->setPeriod(period);
            m_tuner_angular_two->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        std::unique_ptr<Autotuner> m_tuner_two; //!< Autotuner for block size (step two kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_one; //!< Autotuner for block size (angular step one kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_two; //!< Autotuner for block size (angular step two kernel)

    };

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
template<class Manifold>
void TwoStepRATTLENVEGPU<Manifold>::integrateStepOne(unsigned int timestep)
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
template<class Manifold>
void TwoStepRATTLENVEGPU<Manifold>::integrateStepTwo(unsigned int timestep)
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

    gpu_rattle_nve_step_two<Manifold>(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     d_net_force.data,
                     m_manifold,
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


template<class Manifold>
void TwoStepRATTLENVEGPU<Manifold>::IncludeRATTLEForce(unsigned int timestep)
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
    gpu_include_rattle_force_nve<Manifold>(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_net_force.data,
                     d_net_virial.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     net_virial_pitch,
                     m_manifold,
                     m_eta,
                     m_deltaT,
                     m_zero_force,
                     m_tuner_one->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    }

template<class Manifold>
void export_TwoStepRATTLENVEGPU(py::module& m, const std::string& name)
    {
    py::class_<TwoStepRATTLENVEGPU<Manifold>, TwoStepRATTLENVE<Manifold>, std::shared_ptr<TwoStepRATTLENVEGPU<Manifold> > >(m, name.c_str() )
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Manifold>())
        ;
    }

#endif // #ifndef __TWO_STEP_RATTLE_NVE_GPU_H__
