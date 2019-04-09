// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file ZeroMomentumUpdater.cc
    \brief Defines the ZeroMomentumUpdater class
*/


#include "ZeroMomentumUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;
namespace py = pybind11;

/*! \param sysdef System to zero the momentum of
*/
ZeroMomentumUpdater::ZeroMomentumUpdater(std::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing ZeroMomentumUpdater" << endl;
    assert(m_pdata);
    }

ZeroMomentumUpdater::~ZeroMomentumUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying ZeroMomentumUpdater" << endl;
    }

/*! Perform the needed calculations to zero the system's momentum
    \param timestep Current time step of the simulation
*/
void ZeroMomentumUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ZeroMomentum");

    // calculate the average momentum
    assert(m_pdata);

    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;
    unsigned int n = 0;

    // add up the momentum of every free particle (including floppy body particles) and every central particle of a rigid body
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        if (h_body.data[i] >= MIN_FLOPPY || h_body.data[i] == h_tag.data[i])
            {
            Scalar mass = h_vel.data[i].w;
            sum_px += mass*h_vel.data[i].x;
            sum_py += mass*h_vel.data[i].y;
            sum_pz += mass*h_vel.data[i].z;
            n++;
            }
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &sum_px, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &sum_py, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &sum_pz, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    // calculate the average
    Scalar avg_px = sum_px / Scalar(n);
    Scalar avg_py = sum_py / Scalar(n);
    Scalar avg_pz = sum_pz / Scalar(n);

    // subtract this momentum from every free particle (including floppy body particles) and every central particle of a rigid body
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        if (h_body.data[i] >= MIN_FLOPPY || h_body.data[i] == h_tag.data[i])
            {
            Scalar mass = h_vel.data[i].w;
            h_vel.data[i].x -= avg_px/mass;
            h_vel.data[i].y -= avg_py/mass;
            h_vel.data[i].z -= avg_pz/mass;
            }
        }
    } // end GPUArray scope

    if (m_prof) m_prof->pop();
    }

void export_ZeroMomentumUpdater(py::module& m)
    {
    py::class_<ZeroMomentumUpdater, std::shared_ptr<ZeroMomentumUpdater> >(m, "ZeroMomentumUpdater", py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
