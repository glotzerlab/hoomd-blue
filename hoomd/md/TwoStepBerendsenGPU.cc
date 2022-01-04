// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file TwoStepBerendsenGPU.cc
    \brief Defines TwoStepBerendsenGPU
*/

#include "TwoStepBerendsenGPU.h"
#include "TwoStepBerendsenGPU.cuh"

#include <functional>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to which the Berendsen thermostat will be applied
    \param group Group of particles to which the Berendsen thermostat will be applied
    \param thermo Compute for thermodynamic properties
    \param tau Time constant for Berendsen thermostat
    \param T Set temperature
*/
TwoStepBerendsenGPU::TwoStepBerendsenGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<ParticleGroup> group,
                                         std::shared_ptr<ComputeThermo> thermo,
                                         Scalar tau,
                                         std::shared_ptr<Variant> T)
    : TwoStepBerendsen(sysdef, group, thermo, tau, T)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create BerendsenGPU on a CPU device.");
        }

    m_block_size = 256;
    }

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TwoStepBerendsenGPU::integrateStepOne(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (m_prof)
        m_prof->push("Berendsen");

    // compute the current thermodynamic quantities and get the temperature
    m_thermo->compute(timestep);
    Scalar curr_T = m_thermo->getTranslationalTemperature();

    // compute the value of lambda for the current timestep
    Scalar lambda
        = sqrt(Scalar(1.0) + m_deltaT / m_tau * ((*m_T)(timestep) / curr_T - Scalar(1.0)));

    // access the particle data arrays for writing on the GPU
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

    // perform the integration on the GPU
    kernel::gpu_berendsen_step_one(d_pos.data,
                                   d_vel.data,
                                   d_accel.data,
                                   d_image.data,
                                   d_index_array.data,
                                   group_size,
                                   box,
                                   m_block_size,
                                   lambda,
                                   m_deltaT);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop();
    }

void TwoStepBerendsenGPU::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (m_prof)
        m_prof->push("Berendsen");

    // get the net force
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

    // access the particle data arrays for use on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);

    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // perform the second step of the integration on the GPU
    kernel::gpu_berendsen_step_two(d_vel.data,
                                   d_accel.data,
                                   d_index_array.data,
                                   group_size,
                                   d_net_force.data,
                                   m_block_size,
                                   m_deltaT);

    // check if an error occurred
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop();
    }

namespace detail
    {
void export_BerendsenGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepBerendsenGPU, TwoStepBerendsen, std::shared_ptr<TwoStepBerendsenGPU>>(
        m,
        "TwoStepBerendsenGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::shared_ptr<Variant>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
