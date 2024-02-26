// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ExampleUpdater.h"
#ifdef ENABLE_HIP
#include "ExampleUpdater.cuh"
#endif

/*! \file ExampleUpdater.cc
    \brief Definition of ExampleUpdater
*/

// ********************************
// here follows the code for ExampleUpdater on the CPU

namespace hoomd
    {
/*! \param sysdef System to zero the velocities of
 */
ExampleUpdater::ExampleUpdater(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<Trigger> trigger)
    : Updater(sysdef, trigger)
    {
    }

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdater::update(uint64_t timestep)
    {
    Updater::update(timestep);
    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);

    // zero the velocity of every particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        }
    }

namespace detail
    {
/* Export the CPU updater to be visible in the python module
 */
void export_ExampleUpdater(pybind11::module& m)
    {
    pybind11::class_<ExampleUpdater, Updater, std::shared_ptr<ExampleUpdater>>(m, "ExampleUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>());
    }

    } // end namespace detail

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_HIP

/*! \param sysdef System to zero the velocities of
 */
ExampleUpdaterGPU::ExampleUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<Trigger> trigger)
    : ExampleUpdater(sysdef, trigger)
    {
    }

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdaterGPU::update(uint64_t timestep)
    {
    Updater::update(timestep);

    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);

    // call the kernel defined in ExampleUpdater.cu
    kernel::gpu_zero_velocities(d_vel.data, m_pdata->getN());

    // check for error codes from the GPU if error checking is enabled
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

namespace detail
    {
/* Export the GPU updater to be visible in the python module
 */
void export_ExampleUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<ExampleUpdaterGPU, ExampleUpdater, std::shared_ptr<ExampleUpdaterGPU>>(
        m,
        "ExampleUpdaterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>());
    }

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd
