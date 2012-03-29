/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file TempRescaleUpdater.cc
    \brief Defines the TempRescaleUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TempRescaleUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to set temperature on
    \param thermo ComputeThermo to compute the temperature with
    \param tset Temperature set point
*/
TempRescaleUpdater::TempRescaleUpdater(boost::shared_ptr<SystemDefinition> sysdef,
                                       boost::shared_ptr<ComputeThermo> thermo,
                                       boost::shared_ptr<Variant> tset)
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
    Scalar cur_temp = m_thermo->getTemperature();
    
    if (m_prof) m_prof->push("TempRescale");
    
    if (cur_temp < 1e-3)
        {
        m_exec_conf->msg->notice(2) << "update.temp_rescale: cannot scale a 0 temperature to anything but 0, skipping this step" << endl;
        }
    else
        {
        // calculate a fraction to scale the momenta by
        Scalar fraction = sqrt(m_tset->getValue(timestep) / cur_temp);

        // scale the free particle velocities
        assert(m_pdata);
        {
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                    if (h_body.data[i] == NO_BODY)
                    {
                    h_vel.data[i].x *= fraction;
                    h_vel.data[i].y *= fraction;
                    h_vel.data[i].z *= fraction;
                    }
                }
        }

        // scale all the rigid body com velocities and angular momenta
        boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
        unsigned int n_bodies = rigid_data->getNumBodies();
        if (n_bodies > 0)
            {
            ArrayHandle<Scalar4> h_body_vel(rigid_data->getVel(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_body_angmom(rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
            
            for (unsigned int body = 0; body < n_bodies; body++)
                {
                h_body_vel.data[body].x *= fraction;
                h_body_vel.data[body].y *= fraction;
                h_body_vel.data[body].z *= fraction;

                h_body_angmom.data[body].x *= fraction;
                h_body_angmom.data[body].y *= fraction;
                h_body_angmom.data[body].z *= fraction;
                }
            }
        
        // ensure that the particle velocities are up to date
        rigid_data->setRV(false);
        }
    
    if (m_prof) m_prof->pop();
    }

/*! \param tset New temperature set point
    \note The new set point doesn't take effect until the next call to update()
*/
void TempRescaleUpdater::setT(boost::shared_ptr<Variant> tset)
    {
    m_tset = tset;
    }

void export_TempRescaleUpdater()
    {
    class_<TempRescaleUpdater, boost::shared_ptr<TempRescaleUpdater>, bases<Updater>, boost::noncopyable>
    ("TempRescaleUpdater", init< boost::shared_ptr<SystemDefinition>,
                                 boost::shared_ptr<ComputeThermo>,
                                 boost::shared_ptr<Variant> >())
    .def("setT", &TempRescaleUpdater::setT)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

