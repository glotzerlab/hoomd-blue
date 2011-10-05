/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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
    
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    
    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            arrays.ax[j] = arrays.ay[j] = arrays.az[j] = 0.0;
        
        Scalar dx = arrays.vx[j]*m_deltaT + Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT*m_deltaT;
        Scalar dy = arrays.vy[j]*m_deltaT + Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT*m_deltaT;
        Scalar dz = arrays.vz[j]*m_deltaT + Scalar(1.0/2.0)*arrays.az[j]*m_deltaT*m_deltaT;
        
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
            
        arrays.x[j] += dx;
        arrays.y[j] += dy;
        arrays.z[j] += dz;
        
        arrays.vx[j] += Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT;
        arrays.vy[j] += Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT;
        arrays.vz[j] += Scalar(1.0/2.0)*arrays.az[j]*m_deltaT;
        }
    
    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // wrap the particle around the box
        if (arrays.x[j] >= box.xhi)
            {
            arrays.x[j] -= Lx;
            arrays.ix[j]++;
            }
        else if (arrays.x[j] < box.xlo)
            {
            arrays.x[j] += Lx;
            arrays.ix[j]--;
            }
            
        if (arrays.y[j] >= box.yhi)
            {
            arrays.y[j] -= Ly;
            arrays.iy[j]++;
            }
        else if (arrays.y[j] < box.ylo)
            {
            arrays.y[j] += Ly;
            arrays.iy[j]--;
            }
            
        if (arrays.z[j] >= box.zhi)
            {
            arrays.z[j] -= Lz;
            arrays.iz[j]++;
            }
        else if (arrays.z[j] < box.zlo)
            {
            arrays.z[j] += Lz;
            arrays.iz[j]--;
            }
        }
    
    m_pdata->release();
    
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
    
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        if (m_zero_force)
            {
            arrays.ax[j] = arrays.ay[j] = arrays.az[j] = 0.0;
            }
        else
            {
            // first, calculate acceleration from the net force
            Scalar minv = Scalar(1.0) / arrays.mass[j];
            arrays.ax[j] = h_net_force.data[j].x*minv;
            arrays.ay[j] = h_net_force.data[j].y*minv;
            arrays.az[j] = h_net_force.data[j].z*minv;
            }
        
        // then, update the velocity
        arrays.vx[j] += Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT;
        arrays.vy[j] += Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT;
        arrays.vz[j] += Scalar(1.0/2.0)*arrays.az[j]*m_deltaT;
        
        // limit the movement of the particles
        if (m_limit)
            {
            Scalar vel = sqrt(arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
            if ( (vel*m_deltaT) > m_limit_val)
                {
                arrays.vx[j] = arrays.vx[j] / vel * m_limit_val / m_deltaT;
                arrays.vy[j] = arrays.vy[j] / vel * m_limit_val / m_deltaT;
                arrays.vz[j] = arrays.vz[j] / vel * m_limit_val / m_deltaT;
                }
            }
        }
    
    m_pdata->release();
    
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

