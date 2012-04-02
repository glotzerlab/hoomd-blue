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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepNVE.h"

/*! \file TwoStepNVE.h
    \brief Contains code for the TwoStepNVE class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param skip_restart Skip initialization of the restart information
*/
TwoStepNVE::TwoStepNVE(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       bool skip_restart)
    : IntegrationMethodTwoStep(sysdef, group), m_limit(false), m_limit_val(1.0), m_zero_force(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVE" << endl;

    if (!skip_restart)
        {
        // set a named, but otherwise blank set of integrator variables
        IntegratorVariables v = getIntegratorVariables();
        
        if (!restartInfoTestValid(v, "nve", 0))
            {
            v.type = "nve";
            v.variable.resize(0);
            setValidRestart(false);
            }
        else
            setValidRestart(true);

        setIntegratorVariables(v);
        }
    }

TwoStepNVE::~TwoStepNVE()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVE" << endl;
    }

/*! \param limit Distance to limit particle movement each time step

    Once the limit is set, future calls to update() will never move a particle
    a distance larger than the limit in a single time step
*/
void TwoStepNVE::setLimit(Scalar limit)
    {    
    m_limit = true;
    m_limit_val = limit;
    }

/*! Disables the limit, allowing particles to move normally
*/
void TwoStepNVE::removeLimit()
    {
    m_limit = false;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNVE::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    // profile this step
    if (m_prof)
        m_prof->push("NVE step 1");
    
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
        
        Scalar dx = h_vel.data[j].x*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT*m_deltaT;
        Scalar dy = h_vel.data[j].y*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT*m_deltaT;
        Scalar dz = h_vel.data[j].z*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT*m_deltaT;
        
        // limit the movement of the particles
        if (m_limit)
            {
            Scalar len = sqrt(dx*dx + dy*dy + dz*dz);
            if (len > m_limit_val)
                {
                dx = dx / len * m_limit_val;
                dy = dy / len * m_limit_val;
                dz = dz / len * m_limit_val;
                }
            }
            
        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;
        
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        }
    
    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        box.wrap(h_pos.data[j], h_image.data[j]);
        }
    
    // done profiling
    if (m_prof)
        m_prof->pop();
    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVE::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // profile this step
    if (m_prof)
        m_prof->push("NVE step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }
        else
            {
            // first, calculate acceleration from the net force
            Scalar minv = Scalar(1.0) / h_vel.data[j].w;
            h_accel.data[j].x = h_net_force.data[j].x*minv;
            h_accel.data[j].y = h_net_force.data[j].y*minv;
            h_accel.data[j].z = h_net_force.data[j].z*minv;
            }
        
        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        
        // limit the movement of the particles
        if (m_limit)
            {
            Scalar vel = sqrt(h_vel.data[j].x*h_vel.data[j].x+h_vel.data[j].y*h_vel.data[j].y+h_vel.data[j].z*h_vel.data[j].z);
            if ( (vel*m_deltaT) > m_limit_val)
                {
                h_vel.data[j].x = h_vel.data[j].x / vel * m_limit_val / m_deltaT;
                h_vel.data[j].y = h_vel.data[j].y / vel * m_limit_val / m_deltaT;
                h_vel.data[j].z = h_vel.data[j].z / vel * m_limit_val / m_deltaT;
                }
            }
        }
    
    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepNVE()
    {
    class_<TwoStepNVE, boost::shared_ptr<TwoStepNVE>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNVE", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, bool >())
        .def("setLimit", &TwoStepNVE::setLimit)
        .def("removeLimit", &TwoStepNVE::removeLimit)
        .def("setZeroForce", &TwoStepNVE::setZeroForce)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

