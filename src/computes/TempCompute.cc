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

/*! \file TempCompute.cc
    \brief Contains code for the TempCompute class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "TempCompute.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iostream>
using namespace std;


/*! \param sysdef System to compute temperature of

   Note: we have periodic boundary conditions, so we have
   translational invariance, i.e. the number of degrees of
   freedom is 3N-3 (minus constraints when implemented).
*/
TempCompute::TempCompute(boost::shared_ptr<SystemDefinition> sysdef) : Compute(sysdef), m_temp(0.0)
    {
    assert(m_pdata);
    m_dof = m_pdata->getN() * 3 - 3;
    }

/*! Calls computeTemp if the temperature needs updating
    \param timestep Current time step of the simulation
*/
void TempCompute::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep))
        return;
        
    computeTemp();
    }


/*! Computes the temperature by computing the kinetic energy and multiplying by the appropriate factor.
    \note This is computed in reduced units
*/
void TempCompute::computeTemp()
    {
    if (m_prof) m_prof->push("Temp");
    
    assert(m_pdata);
    assert(m_dof != 0);
    
    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    
    // total up kinetic energy
    Scalar K = 0.0;
    // K = Sum(m * v**2)
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        K += arrays.mass[i] * (arrays.vx[i] * arrays.vx[i] + arrays.vy[i] * arrays.vy[i] + arrays.vz[i]*arrays.vz[i]);
        }
        
    // K = 1/2 * k_b * T * dof
    // => T = K * 2 / dof / k_b
    // but the variable K is already K*2
    m_temp = K / Scalar(m_dof);
    
    m_pdata->release();
    
    if (m_prof) m_prof->pop();
    }

void export_TempCompute()
    {
    class_<TempCompute, boost::shared_ptr<TempCompute>, bases<Compute>, boost::noncopyable >
    ("TempCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setDOF", &TempCompute::setDOF)
    .def("getTemp", &TempCompute::getTemp)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

