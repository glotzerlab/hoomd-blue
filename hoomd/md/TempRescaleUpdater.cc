// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TempRescaleUpdater.cc
    \brief Defines the TempRescaleUpdater class
*/


#include "TempRescaleUpdater.h"

namespace py = pybind11;

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to set temperature on
    \param thermo ComputeThermo to compute the temperature with
    \param tset Temperature set point
*/
TempRescaleUpdater::TempRescaleUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ComputeThermo> thermo,
                                       std::shared_ptr<Variant> tset)
        : Updater(sysdef), m_thermo(thermo), m_tset(tset)
    {
    m_exec_conf->msg->notice(5) << "Constructing TempRescaleUpdater" << endl;

    assert(m_pdata);
    assert(thermo);
    }

TempRescaleUpdater::~TempRescaleUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying TempRescaleUpdater" << endl;
    }


/*! Perform the proper velocity rescaling
    \param timestep Current time step of the simulation
*/
void TempRescaleUpdater::update(unsigned int timestep)
    {
    // find the current temperature

    assert(m_thermo);
    m_thermo->compute(timestep);
    Scalar cur_temp = m_thermo->getTranslationalTemperature();

    if (m_prof) m_prof->push("TempRescale");

    if (cur_temp < 1e-3)
        {
        m_exec_conf->msg->notice(2) << "update.temp_rescale: cannot scale a 0 translational temperature to anything but 0, skipping this step" << endl;
        }
    else
        {
        // calculate a fraction to scale the momenta by
        Scalar fraction = sqrt(m_tset->getValue(timestep) / cur_temp);

        // scale the free particle velocities
        assert(m_pdata);
            {
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                h_vel.data[i].x *= fraction;
                h_vel.data[i].y *= fraction;
                h_vel.data[i].z *= fraction;
                }
            }

        }

    cur_temp = m_thermo->getRotationalTemperature();
    if (! std::isnan(cur_temp))
        {
        // only rescale if we have rotational degrees of freedom
        if (cur_temp < 1e-3)
            {
            m_exec_conf->msg->notice(2) << "update.temp_rescale: cannot scale a 0 rotational temperature to anything but 0, skipping this step" << endl;
            }
        else
            {
            // calculate a fraction to scale the momenta by
            Scalar fraction = sqrt(m_tset->getValue(timestep) / cur_temp);

            // scale the free particle velocities
            assert(m_pdata);
                {
                ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);

                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    h_angmom.data[i].x *= fraction;
                    h_angmom.data[i].y *= fraction;
                    h_angmom.data[i].z *= fraction;
                    }
                }

            }
        }

    if (m_prof) m_prof->pop();
    }

/*! \param tset New temperature set point
    \note The new set point doesn't take effect until the next call to update()
*/
void TempRescaleUpdater::setT(std::shared_ptr<Variant> tset)
    {
    m_tset = tset;
    }

void export_TempRescaleUpdater(py::module& m)
    {
    py::class_<TempRescaleUpdater, std::shared_ptr<TempRescaleUpdater> >(m, "TempRescaleUpdater", py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                                 std::shared_ptr<ComputeThermo>,
                                 std::shared_ptr<Variant> >())
    .def("setT", &TempRescaleUpdater::setT)
    ;
    }
