// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ExampleUpdater.h"
#ifdef ENABLE_CUDA
#include "ExampleUpdater.cuh"
#endif

/*! \file ExampleUpdater.cc
    \brief Definition of ExampleUpdater
*/

// ********************************
// here follows the code for ExampleUpdater on the CPU

/*! \param sysdef System to zero the velocities of
*/
ExampleUpdater::ExampleUpdater(std::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ExampleUpdater");

    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

    // zero the velocity of every particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        }

    if (m_prof) m_prof->pop();
    }

/* Export the CPU updater to be visible in the python module
 */
void export_ExampleUpdater(pybind11::module& m)
    {
    pybind11::class_<ExampleUpdater, std::shared_ptr<ExampleUpdater> >(m, "ExampleUpdater", pybind11::base<Updater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition> >())
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
ExampleUpdaterGPU::ExampleUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef)
        : ExampleUpdater(sysdef)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdaterGPU::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ExampleUpdater");

    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);

    // call the kernel defined in ExampleUpdater.cu
    gpu_zero_velocities(d_vel.data, m_pdata->getN());

    // check for error codes from the GPU if error checking is enabled
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop();
    }

/* Export the GPU updater to be visible in the python module
 */
void export_ExampleUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<ExampleUpdaterGPU, std::shared_ptr<ExampleUpdaterGPU> >(m, "ExampleUpdaterGPU", pybind11::base<ExampleUpdater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition> >())
    ;
    }

#endif // ENABLE_CUDA
