/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

/*! \file Enforce2DUpdater.cc
    \brief Defines the Enforce2DUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "Enforce2DUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to zero the momentum of
*/
Enforce2DUpdater::Enforce2DUpdater(boost::shared_ptr<SystemDefinition> sysdef)
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

void export_Enforce2DUpdater()
    {
    class_<Enforce2DUpdater, boost::shared_ptr<Enforce2DUpdater>, bases<Updater>, boost::noncopyable>
    ("Enforce2DUpdater", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
