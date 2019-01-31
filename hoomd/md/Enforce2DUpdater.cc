// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file Enforce2DUpdater.cc
    \brief Defines the Enforce2DUpdater class
*/


#include "Enforce2DUpdater.h"

namespace py = pybind11;

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to zero the momentum of
*/
Enforce2DUpdater::Enforce2DUpdater(std::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing Enforce2DUpdater" << endl;

    assert(m_pdata);
    if (m_sysdef->getNDimensions() != 2)
        {
        m_exec_conf->msg->error() << "update.enforce2d: used for 3 dimensional system" << endl;
        throw runtime_error("Error initializing Enforce2DUpdater");
        }
    }

Enforce2DUpdater::~Enforce2DUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying Enforce2DUpdater" << endl;
    }

/*! Perform the needed calculations to zero the system's momentum
    \param timestep Current time step of the simulation
*/
void Enforce2DUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Enforce2D");

    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    // zero the z-velocities and z-accelerations:
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_vel.data[i].z = Scalar(0.0);
        h_accel.data[i].z = Scalar(0.0);
        }

    // for rigid bodies, zero x / y components of omega/angmom/torque:

    if (m_prof) m_prof->pop();
    }

void export_Enforce2DUpdater(py::module& m)
    {
    py::class_<Enforce2DUpdater, std::shared_ptr<Enforce2DUpdater> >(m, "Enforce2DUpdater", py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
