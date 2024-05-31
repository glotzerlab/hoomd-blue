// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file BounceBackNVEGPU.h
 * \brief Declares the BounceBackNVEGPU class for doing NVE integration with bounce-back
 *        boundary conditions imposed by a geometry using the GPU.
 */

#ifndef MPCD_BOUNCE_BACK_NVE_GPU_H_
#define MPCD_BOUNCE_BACK_NVE_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BounceBackNVE.h"
#include "BounceBackNVEGPU.cuh"

namespace hoomd
    {
namespace mpcd
    {
//! Integrator that applies bounce-back boundary conditions in NVE using the GPU.
/*!
 * See BounceBackNVE for more details.
 */
template<class Geometry> class PYBIND11_EXPORT BounceBackNVEGPU : public BounceBackNVE<Geometry>
    {
    public:
    //! Constructor
    BounceBackNVEGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<const Geometry> geom)
        : BounceBackNVE<Geometry>(sysdef, group, geom)
        {
        m_tuner_1.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "nve_bounce_1"));
        m_tuner_2.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "nve_bounce_2"));
        this->m_autotuners.insert(this->m_autotuners.end(), {m_tuner_1, m_tuner_2});
        }

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    private:
    std::shared_ptr<Autotuner<1>> m_tuner_1;
    std::shared_ptr<Autotuner<1>> m_tuner_2;
    };

template<class Geometry> void BounceBackNVEGPU<Geometry>::integrateStepOne(uint64_t timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "mpcd.integrate: anisotropic particles are not "
                                           "supported with bounce-back integrators."
                                        << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with bounce-back");
        }

    if (this->m_validate_geom)
        this->validate();

    // particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::read);
    const BoxDim box = this->m_pdata->getBox();

    // group members
    const unsigned int group_size = this->m_group->getNumMembers();
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(),
                                      access_location::device,
                                      access_mode::read);

    this->m_tuner_1->begin();
    gpu::bounce_args_t args(d_pos.data,
                            d_image.data,
                            d_vel.data,
                            d_accel.data,
                            d_group.data,
                            this->m_deltaT,
                            box,
                            group_size,
                            this->m_tuner_1->getParam()[0]);

    gpu::nve_bounce_step_one<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner_1->end();
    }

template<class Geometry> void BounceBackNVEGPU<Geometry>::integrateStepTwo(uint64_t timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "mpcd.integrate: anisotropic particles are not "
                                           "supported with bounce-back integrators."
                                        << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with bounce-back");
        }

    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_force(this->m_pdata->getNetForce(),
                                     access_location::device,
                                     access_mode::read);

    const unsigned int group_size = this->m_group->getNumMembers();
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(),
                                      access_location::device,
                                      access_mode::read);

    this->m_tuner_2->begin();
    gpu::nve_bounce_step_two(d_vel.data,
                             d_accel.data,
                             d_net_force.data,
                             d_group.data,
                             this->m_deltaT,
                             group_size,
                             this->m_tuner_2->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner_2->end();
    }

namespace detail
    {
//! Exports the BounceBackNVEGPU class to python
template<class Geometry> void export_BounceBackNVEGPU(pybind11::module& m)
    {
    const std::string name = "BounceBackNVE" + Geometry::getName() + "GPU";

    pybind11::class_<BounceBackNVEGPU<Geometry>,
                     BounceBackNVE<Geometry>,
                     std::shared_ptr<BounceBackNVEGPU<Geometry>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<const Geometry>>());
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_BOUNCE_BACK_NVE_GPU_H_
