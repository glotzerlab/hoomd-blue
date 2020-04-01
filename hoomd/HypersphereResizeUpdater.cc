// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file SphereResizeUpdater.cc
    \brief Defines the SphereResizeUpdater class
*/


#include "HypersphereResizeUpdater.h"

#include "Hypersphere.h"

#include <math.h>
#include <iostream>
#include <stdexcept>

using namespace std;
namespace py = pybind11;

/*! \param sysdef System definition containing the particle data to set the hypersphere size on
    \param R radius of the hypersphere over time

    The default setting is to scale particle positions along with the hypersphere.
*/
HypersphereResizeUpdater::HypersphereResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Variant> R)
    : Updater(sysdef), m_R(R)
    {
    assert(m_pdata);
    assert(m_R);

    m_exec_conf->msg->notice(5) << "Constructing HypersphereResizeUpdater" << endl;
    }

HypersphereResizeUpdater::~HypersphereResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying HypersphereResizeUpdater" << endl;
    }

/*! Rescales the simulation hypersphere
    \param timestep Current time step of the simulation
*/
void HypersphereResizeUpdater::update(unsigned int timestep)
    {
    m_exec_conf->msg->notice(10) << "Hypersphere resize update" << endl;
    if (m_prof) m_prof->push("HypersphereResize");

    // first, compute what the current sphere size and tilt factors should be
    Scalar R = m_R->getValue(timestep);

    Hypersphere hypersphere = m_pdata->getHypersphere();
    hypersphere.setR(R);

    // set the new hypersphere
    m_pdata->setHypersphere(hypersphere);

    if (m_prof) m_prof->pop();
    }

void export_HypersphereResizeUpdater(py::module& m)
    {
    py::class_<HypersphereResizeUpdater, std::shared_ptr<HypersphereResizeUpdater> >(m,"HypersphereResizeUpdater",py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
     std::shared_ptr<Variant> >());
    }
