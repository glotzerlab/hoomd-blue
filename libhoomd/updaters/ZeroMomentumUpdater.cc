/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// $Id$
// $URL$
// Maintainer: joaander

/*! \file ZeroMomentumUpdater.cc
    \brief Defines the ZeroMomentumUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "ZeroMomentumUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to zero the momentum of
*/
ZeroMomentumUpdater::ZeroMomentumUpdater(boost::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    assert(m_pdata);
    }


/*! Perform the needed calculations to zero the system's momentum
    \param timestep Current time step of the simulation
*/
void ZeroMomentumUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ZeroMomentum");
    
    // calculate the average momentum
    assert(m_pdata);
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    
    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;
    unsigned int n = 0;
    
    // add up the momentum of every free particle
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        if (arrays.body[i] == NO_BODY)
            {
            Scalar mass = arrays.mass[i];
            sum_px += mass*arrays.vx[i];
            sum_py += mass*arrays.vy[i];
            sum_pz += mass*arrays.vz[i];
            n++;
            }
        }

    // add up the linear momentum of all bodies
    boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
    unsigned int n_bodies = rigid_data->getNumBodies();
    if (n_bodies > 0)
        {
        ArrayHandle<Scalar4> h_body_vel(rigid_data->getVel(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_body_mass(rigid_data->getBodyMass(), access_location::host, access_mode::read);
        
        for (unsigned int body = 0; body < n_bodies; body++)
            {
            Scalar mass = h_body_mass.data[body];
            Scalar4 vel = h_body_vel.data[body];
            sum_px += mass * vel.x;
            sum_py += mass * vel.y;
            sum_pz += mass * vel.z;
            n++;
            }
        }
    
    // calculate the average
    Scalar avg_px = sum_px / Scalar(n);
    Scalar avg_py = sum_py / Scalar(n);
    Scalar avg_pz = sum_pz / Scalar(n);
    
    // subtract this momentum from every free partcile
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        if (arrays.body[i] == NO_BODY)
            {
            Scalar mass = arrays.mass[i];
            arrays.vx[i] -= avg_px/mass;
            arrays.vy[i] -= avg_py/mass;
            arrays.vz[i] -= avg_pz/mass;
            }
        }
        
    // subtract this momentum from every rigid body
    if (n_bodies > 0)
        {
        ArrayHandle<Scalar4> h_body_vel(rigid_data->getVel(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_body_mass(rigid_data->getBodyMass(), access_location::host, access_mode::read);
        
        for (unsigned int body = 0; body < n_bodies; body++)
            {
            Scalar mass = h_body_mass.data[body];
            h_body_vel.data[body].x -= avg_px/mass;
            h_body_vel.data[body].y -= avg_py/mass;
            h_body_vel.data[body].z -= avg_pz/mass;
            }
        }

    m_pdata->release();
    
    // update the body particle velocities to reflect the new body velocities
    rigid_data->setRV(false);
    
    if (m_prof) m_prof->pop();
    }

void export_ZeroMomentumUpdater()
    {
    class_<ZeroMomentumUpdater, boost::shared_ptr<ZeroMomentumUpdater>, bases<Updater>, boost::noncopyable>
    ("ZeroMomentumUpdater", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

