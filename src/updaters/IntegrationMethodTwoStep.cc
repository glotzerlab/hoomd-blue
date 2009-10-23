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

#include "IntegrationMethodTwoStep.h"

/*! \file IntegrationMethodTwoStep.h
    \brief Contains code for the IntegrationMethodTwoStep class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \post The method is constructed with the given particle data and a NULL profiler.
*/
IntegrationMethodTwoStep::IntegrationMethodTwoStep(boost::shared_ptr<SystemDefinition> sysdef,
                                                   boost::shared_ptr<ParticleGroup> group)
    : m_sysdef(sysdef), m_group(group), m_pdata(m_sysdef->getParticleData()), exec_conf(m_pdata->getExecConf()), 
      m_deltaT(Scalar(0.0))
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    assert(m_group);
    }

/*! It is useful for the user to know where computation time is spent, so all integration methods
    should profile themselves. This method sets the profiler for them to use.
    This method does not need to be called, as Computes will not profile themselves
    on a NULL profiler
    \param prof Pointer to a profiler for the compute to use. Set to NULL
        (boost::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void IntegrationMethodTwoStep::setProfiler(boost::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

/*! \param deltaT New time step to set
*/
void IntegrationMethodTwoStep::setDeltaT(Scalar deltaT)
    {
    m_deltaT = deltaT;
    }

void export_IntegrationMethodTwoStep()
    {
    class_<IntegrationMethodTwoStep, boost::shared_ptr<IntegrationMethodTwoStep>, boost::noncopyable>
        ("IntegrationMethodTwoStep", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup> >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
