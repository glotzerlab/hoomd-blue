// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Enforce2DUpdaterGPU.cc
    \brief Defines the Enforce2DUpdaterGPU class
*/


#include "Enforce2DUpdaterGPU.h"
#include "Enforce2DUpdaterGPU.cuh"

namespace py = pybind11;

using namespace std;

/*! \param sysdef System to update
*/
Enforce2DUpdaterGPU::Enforce2DUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef) : Enforce2DUpdater(sysdef)
    {
    // at least one GPU is needed
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a Enforce2DUpdaterGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing Enforce2DUpdaterGPU");
        }
    }

/*! \param timestep Current time step of the simulation

    Calls gpu_enforce2d to do the actual work.
*/
void Enforce2DUpdaterGPU::update(unsigned int timestep)
    {
    assert(m_pdata);

    if (m_prof)
        m_prof->push(m_exec_conf, "Enforce2D");

    // access the particle data arrays
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    // call the enforce 2d kernel
    gpu_enforce2d(m_pdata->getN(),
                  d_vel.data,
                  d_accel.data);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_Enforce2DUpdaterGPU(py::module& m)
    {
    py::class_<Enforce2DUpdaterGPU, std::shared_ptr<Enforce2DUpdaterGPU> >(m, "Enforce2DUpdaterGPU", py::base<Enforce2DUpdater>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
