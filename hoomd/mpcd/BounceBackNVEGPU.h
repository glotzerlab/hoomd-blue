// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file BounceBackNVEGPU.h
 * \brief Declares the BounceBackNVEGPU class for doing NVE integration with bounce-back
 *        boundary conditions imposed by a geometry using the GPU.
 */

#ifndef MPCD_BOUNCE_BACK_NVE_GPU_H_
#define MPCD_BOUNCE_BACK_NVE_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "BounceBackNVE.h"
#include "BounceBackNVEGPU.cuh"

namespace mpcd
{
//! Integrator that applies bounce-back boundary conditions in NVE using the GPU.
/*!
 * See BounceBackNVE for more details.
 */
template<class Geometry>
class PYBIND11_EXPORT BounceBackNVEGPU : public BounceBackNVE<Geometry>
    {
    public:
        //! Constructor
        BounceBackNVEGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         std::shared_ptr<const Geometry> geom)
            : BounceBackNVE<Geometry>(sysdef, group, geom)
            {
            m_tuner_1.reset(new Autotuner(32, 1024, 32, 5, 100000, "nve_bounce_1", this->m_exec_conf));
            m_tuner_2.reset(new Autotuner(32, 1024, 32, 5, 100000, "nve_bounce_2", this->m_exec_conf));
            }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         *
         * Derived classes should override this to set the parameters of their autotuners.
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            BounceBackNVE<Geometry>::setAutotunerParams(enable, period);
            m_tuner_1->setEnabled(enable); m_tuner_1->setPeriod(period);
            m_tuner_2->setEnabled(enable); m_tuner_2->setPeriod(period);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner_1;
        std::unique_ptr<Autotuner> m_tuner_2;
    };

template<class Geometry>
void BounceBackNVEGPU<Geometry>::integrateStepOne(unsigned int timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "mpcd.integrate: anisotropic particles are not supported with bounce-back integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with bounce-back");
        }
    if (this->m_prof) this->m_prof->push("Bounce NVE step 1");

    if (this->m_validate_geom) this->validate();

    // particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(), access_location::device, access_mode::read);
    const BoxDim& box = this->m_pdata->getBox();

    // group members
    const unsigned int group_size = this->m_group->getNumMembers();
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(), access_location::device, access_mode::read);

    this->m_tuner_1->begin();
    gpu::bounce_args_t args(d_pos.data,
                            d_image.data,
                            d_vel.data,
                            d_accel.data,
                            d_group.data,
                            this->m_deltaT,
                            box,
                            group_size,
                            this->m_tuner_1->getParam());

    gpu::nve_bounce_step_one<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    this->m_tuner_1->end();

    if (this->m_prof) this->m_prof->pop();
    }

template<class Geometry>
void BounceBackNVEGPU<Geometry>::integrateStepTwo(unsigned int timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "mpcd.integrate: anisotropic particles are not supported with bounce-back integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with bounce-back");
        }
    if (this->m_prof) this->m_prof->push("Bounce NVE step 2");

    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_force(this->m_pdata->getNetForce(), access_location::device, access_mode::read);

    const unsigned int group_size = this->m_group->getNumMembers();
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(), access_location::device, access_mode::read);

    this->m_tuner_2->begin();
    gpu::nve_bounce_step_two(d_vel.data,
                             d_accel.data,
                             d_net_force.data,
                             d_group.data,
                             this->m_deltaT,
                             group_size,
                             this->m_tuner_2->getParam());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    this->m_tuner_2->end();

    if (this->m_prof) this->m_prof->pop();
    }

namespace detail
{
//! Exports the BounceBackNVEGPU class to python
template<class Geometry>
void export_BounceBackNVEGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = "BounceBackNVE" + Geometry::getName() + "GPU";

    py::class_<BounceBackNVEGPU<Geometry>, std::shared_ptr<BounceBackNVEGPU<Geometry>>>
        (m, name.c_str(), py::base<BounceBackNVE<Geometry>>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<const Geometry>>())
        ;
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_BOUNCE_BACK_NVE_GPU_H_
