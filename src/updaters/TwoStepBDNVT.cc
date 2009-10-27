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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepBDNVT.h"

/*! \file TwoStepBDNVT.h
    \brief Contains code for the TwoStepBDNVT class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param gamma_diam Set gamma to the particle diameter of each particle if true, otherwise use a per-type
                      gamma via setGamma()
*/
TwoStepBDNVT::TwoStepBDNVT(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool gamma_diam)
    : TwoStepNVE(sysdef, group), m_T(T), m_seed(seed), m_saru(seed), m_gamma_diam(gamma_diam)
    {
    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GPUArray<Scalar> gamma(m_pdata->getNTypes(), m_pdata->getExecConf());
    m_gamma.swap(gamma);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.getNumElements(); i++)
        h_gamma.data[i] = Scalar(1.0);
    }

/*! \param typ Particle type to set gamma for
    \param gamma The gamma value to set
*/
void TwoStepBDNVT::setGamma(unsigned int typ, Scalar gamma)
    {
    // check for user errors
    if (m_gamma_diam)
        {
        cerr << endl << "***Error! Trying to set gamma when it is set to be the diameter! " << typ << endl << endl;
        throw runtime_error("Error setting params in TwoStepBDNVT");
        }
    if (typ >= m_pdata->getNTypes())
        {
        cerr << endl << "***Error! Trying to set gamma for a non existant type! " << typ << endl << endl;
        throw runtime_error("Error setting params in TwoStepBDNVT");
        }
        
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    h_gamma.data[typ] = gamma;
    }

/*! \param timestep Current time step
    \param net_force The net force on each particle in the system
    \param net_virial The net virial on each particle in the system
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepBDNVT::integrateStepTwo(unsigned int timestep,
                                    const GPUArray< Scalar4 >& net_force,
                                    const GPUArray< Scalar >& net_virial)
    {
    assert(!net_force.isNull());
    assert(!net_virial.isNull());
    
    // profile this step
    if (m_prof)
        m_prof->push("NVE step 2");
    
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    
    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    
    // a(t+deltaT) gets modified with the bd forces
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    unsigned int group_size = m_group->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        // first, calculate the BD forces
        // Generate three random numbers
        Scalar rx = m_saru.d(-1,1);
        Scalar ry = m_saru.d(-1,1);
        Scalar rz =  m_saru.d(-1,1);
        
        Scalar gamma;
        if (m_gamma_diam)
            gamma = arrays.diameter[j];
        else
            gamma = h_gamma.data[arrays.type[j]];
        
        // compute the bd force
        Scalar coeff = sqrt(Scalar(6.0)*gamma*currentTemp/m_deltaT);
        Scalar bd_fx = rx*coeff - gamma*arrays.vx[j];
        Scalar bd_fy = ry*coeff - gamma*arrays.vy[j];
        Scalar bd_fz = rz*coeff - gamma*arrays.vz[j];
        
        // then, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / arrays.mass[j];
        arrays.ax[j] = (h_net_force.data[j].x + bd_fx)*minv;
        arrays.ay[j] = (h_net_force.data[j].y + bd_fy)*minv;
        arrays.az[j] = (h_net_force.data[j].z + bd_fz)*minv;
        
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

void export_TwoStepBDNVT()
    {
    class_<TwoStepBDNVT, boost::shared_ptr<TwoStepBDNVT>, boost::noncopyable>
        ("TwoStepBDNVT", init< boost::shared_ptr<SystemDefinition>,
                         boost::shared_ptr<ParticleGroup>,
                         boost::shared_ptr<Variant>,
                         unsigned int,
                         bool
                         >())
        .def("setT", &TwoStepBDNVT::setT)
        .def("setGamma", &TwoStepBDNVT::setGamma)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

