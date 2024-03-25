// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __TWO_STEP_RATTLE_NVE_GPU_H__
#define __TWO_STEP_RATTLE_NVE_GPU_H__

#ifdef ENABLE_HIP

/*! \file TwoStepRATTLENVEGPU.h
\brief Declares the TwoStepRATTLENVEGPU class
*/

#include "hoomd/md/TwoStepRATTLENVE.h"
#include "hoomd/md/TwoStepRATTLENVEGPU.cuh"

#include "hoomd/Autotuner.h"
#include <utility>

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Integrates part of the system forward in two steps in the NVE ensemble on the GPU
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface, runs
on the GPU \ingroup updaters
*/
template<class Manifold>
class PYBIND11_EXPORT TwoStepRATTLENVEGPU : public TwoStepRATTLENVE<Manifold>
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepRATTLENVEGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ParticleGroup> group,
                        Manifold manifold,
                        Scalar tolerance);

    virtual ~TwoStepRATTLENVEGPU() {};

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Includes the RATTLE forces to the virial/net force
    virtual void includeRATTLEForce(uint64_t timestep);

    std::pair<bool, Scalar> getKernelLimitValues(uint64_t timestep)
        {
        auto use_limit = !(this->m_limit == nullptr);
        Scalar maximum_displacement;
        if (use_limit)
            {
            maximum_displacement = this->m_limit->operator()(timestep);
            }
        else
            {
            maximum_displacement = 0.0;
            }
        return std::pair<bool, Scalar>(use_limit, maximum_displacement);
        }

    private:
    /// Autotuner for block size (step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_one;

    /// Autotuner for block size (step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_two;

    /// Autotuner for block size (force kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_force;

    /// Autotuner for block size (angular step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_one;

    /// Autotuner for block size (angular step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_two;
    };

/*! \file TwoStepRATTLENVEGPU.h
    \brief Contains code for the TwoStepRATTLENVEGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
*/
template<class Manifold>
TwoStepRATTLENVEGPU<Manifold>::TwoStepRATTLENVEGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<ParticleGroup> group,
                                                   Manifold manifold,
                                                   Scalar tolerance)
    : TwoStepRATTLENVE<Manifold>(sysdef, group, manifold, tolerance)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a TwoStepRATTLENVEGPU when CUDA is disabled" << std::endl;
        throw std::runtime_error("Error initializing TwoStepRATTLENVEGPU");
        }

    m_tuner_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                       this->m_exec_conf,
                                       "rattle_nve_step_one"));
    m_tuner_two.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                       this->m_exec_conf,
                                       "rattle_nve_step_two"));
    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "rattle_nve_force"));
    m_tuner_angular_one.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "rattle_nve_angular_one",
                         5,
                         true));
    m_tuner_angular_two.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "rattle_nve_angular_two",
                         5,
                         true));
    this->m_autotuners.insert(
        this->m_autotuners.end(),
        {m_tuner_one, m_tuner_two, m_tuner_force, m_tuner_angular_one, m_tuner_angular_two});
    }
/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.
*/
template<class Manifold> void TwoStepRATTLENVEGPU<Manifold>::integrateStepOne(uint64_t timestep)
    {
    // access all the needed data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    if (this->m_box_changed)
        {
        if (!this->m_manifold.fitsInsideBox(this->m_pdata->getGlobalBox()))
            {
            throw std::runtime_error("Parts of the manifold are outside the box");
            }
        this->m_box_changed = false;
        }

    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // perform the update on the GPU
    auto limit_params = this->getKernelLimitValues(timestep);

    this->m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    kernel::gpu_rattle_nve_step_one(d_pos.data,
                                    d_vel.data,
                                    d_accel.data,
                                    d_image.data,
                                    d_index_array.data,
                                    this->m_group->getGPUPartition(),
                                    this->m_pdata->getBox(),
                                    this->m_deltaT,
                                    limit_params.first,
                                    limit_params.second,
                                    m_tuner_one->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_one->end();
    this->m_exec_conf->endMultiGPU();

    if (this->m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(this->m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(this->m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(this->m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        this->m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        kernel::gpu_rattle_nve_angular_step_one(d_orientation.data,
                                                d_angmom.data,
                                                d_inertia.data,
                                                d_net_torque.data,
                                                d_index_array.data,
                                                this->m_group->getGPUPartition(),
                                                this->m_deltaT,
                                                1.0,
                                                m_tuner_angular_one->getParam()[0]);

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_one->end();
        this->m_exec_conf->endMultiGPU();
        }
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
template<class Manifold> void TwoStepRATTLENVEGPU<Manifold>::integrateStepTwo(uint64_t timestep)
    {
    const GlobalArray<Scalar4>& net_force = this->m_pdata->getNetForce();

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // perform the update on the GPU
    this->m_exec_conf->beginMultiGPU();
    m_tuner_two->begin();

    auto limit_params = this->getKernelLimitValues(timestep);

    kernel::gpu_rattle_nve_step_two<Manifold>(d_pos.data,
                                              d_vel.data,
                                              d_accel.data,
                                              d_index_array.data,
                                              this->m_group->getGPUPartition(),
                                              d_net_force.data,
                                              this->m_manifold,
                                              this->m_tolerance,
                                              this->m_deltaT,
                                              limit_params.first,
                                              limit_params.second,
                                              this->m_zero_force,
                                              m_tuner_two->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_two->end();
    this->m_exec_conf->endMultiGPU();

    if (this->m_aniso)
        {
        // second part of angular update
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_angmom(this->m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(this->m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(this->m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        this->m_exec_conf->beginMultiGPU();
        m_tuner_angular_two->begin();

        kernel::gpu_rattle_nve_angular_step_two(d_orientation.data,
                                                d_angmom.data,
                                                d_inertia.data,
                                                d_net_torque.data,
                                                d_index_array.data,
                                                this->m_group->getGPUPartition(),
                                                this->m_deltaT,
                                                1.0,
                                                m_tuner_angular_two->getParam()[0]);

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_two->end();
        this->m_exec_conf->endMultiGPU();
        }
    }

template<class Manifold> void TwoStepRATTLENVEGPU<Manifold>::includeRATTLEForce(uint64_t timestep)
    {
    // access all the needed data
    const GlobalArray<Scalar4>& net_force = this->m_pdata->getNetForce();
    const GlobalArray<Scalar>& net_virial = this->m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    size_t net_virial_pitch = net_virial.getPitch();

    // perform the update on the GPU
    this->m_exec_conf->beginMultiGPU();
    m_tuner_force->begin();
    kernel::gpu_include_rattle_force_nve<Manifold>(d_pos.data,
                                                   d_vel.data,
                                                   d_accel.data,
                                                   d_net_force.data,
                                                   d_net_virial.data,
                                                   d_index_array.data,
                                                   this->m_group->getGPUPartition(),
                                                   net_virial_pitch,
                                                   this->m_manifold,
                                                   this->m_tolerance,
                                                   this->m_deltaT,
                                                   this->m_zero_force,
                                                   m_tuner_force->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_force->end();
    this->m_exec_conf->endMultiGPU();
    }

namespace detail
    {
template<class Manifold>
void export_TwoStepRATTLENVEGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<TwoStepRATTLENVEGPU<Manifold>,
                     TwoStepRATTLENVE<Manifold>,
                     std::shared_ptr<TwoStepRATTLENVEGPU<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold,
                            Scalar>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // #ifndef __TWO_STEP_RATTLE_NVE_GPU_H__
