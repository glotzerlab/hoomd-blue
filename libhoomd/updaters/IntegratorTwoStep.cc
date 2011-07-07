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

/*! \file IntegratorTwoStep.cc
    \brief Defines the IntegratorTwoStep class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "IntegratorTwoStep.h"

#include <boost/bind.hpp>
using namespace boost;

IntegratorTwoStep::IntegratorTwoStep(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : Integrator(sysdef, deltaT), m_first_step(true), m_prepared(false), m_gave_warning(false)
    {
    }

IntegratorTwoStep::~IntegratorTwoStep()
    {
    }

/*! \param prof The profiler to set
    Sets the profiler both for this class and all of the containted integration methods
*/
void IntegratorTwoStep::setProfiler(boost::shared_ptr<Profiler> prof)
    {
    Integrator::setProfiler(prof);
    
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->setProfiler(prof);
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > IntegratorTwoStep::getProvidedLogQuantities()
    {
    std::vector<std::string> combined_result;
    std::vector<std::string> result;
    
    // Get base class provided log quantities
    result = Integrator::getProvidedLogQuantities();
    combined_result.insert(combined_result.end(), result.begin(), result.end());
    
    // add integrationmethod quantities
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        result = (*method)->getProvidedLogQuantities();
        combined_result.insert(combined_result.end(), result.begin(), result.end());
        }
    return combined_result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
*/
Scalar IntegratorTwoStep::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    bool quantity_flag = false;
    Scalar log_value;

    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        log_value = (*method)->getLogValue(quantity,timestep,quantity_flag);
        if (quantity_flag) return log_value;
        }
    return Integrator::getLogValue(quantity, timestep);
    }

/*! \param timestep Current time step of the simulation
    \post All integration methods previously added with addIntegrationMethod() are applied in order to move the system
          state variables forward to \a timestep+1.
    \post Internally, all forces added via Integrator::addForceCompute are evaluated at \a timestep+1
*/
void IntegratorTwoStep::update(unsigned int timestep)
    {
    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0)
        {
        cout << "***Warning! No integration methods are set, continuing anyways." << endl;
        m_gave_warning = true;
        }
    
    // ensure that prepRun() has been called
    assert(m_prepared);
    
    if (m_prof)
        m_prof->push("Integrate");
    
    // perform the first step of the integration on all groups
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepOne(timestep);
    
    if (m_prof)
        m_prof->pop();
    
    // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);
    
    if (m_prof)
        m_prof->push("Integrate");
    
    // perform the second step of the integration on all groups
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);
    
    if (m_prof)
        m_prof->pop();
    }

/*! \param deltaT new deltaT to set
    \post \a deltaT is also set on all contained integration methods
*/
void IntegratorTwoStep::setDeltaT(Scalar deltaT)
    {
    Integrator::setDeltaT(deltaT);
    
    // set deltaT on all methods already added
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->setDeltaT(deltaT);
    }

/*! \param new_method New integration method to add to the integrator
    Before the method is added, it is checked to see if the group intersects with any of the groups integrated by
    existing methods. If an interesection is found, an error is issued. If no interesection is found, setDeltaT
    is called on the method and it is added to the list.
*/
void IntegratorTwoStep::addIntegrationMethod(boost::shared_ptr<IntegrationMethodTwoStep> new_method)
    {
    // check for intersections with existing methods
    shared_ptr<ParticleGroup> new_group = new_method->getGroup();
    
    if (new_group->getNumMembers() == 0)
        cout << "***Warning! An integration method has been added that operates on zero particles." << endl;
    
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        shared_ptr<ParticleGroup> intersection = ParticleGroup::groupIntersection(new_group, current_group);
        
        if (intersection->getNumMembers() > 0)
            {
            cerr << endl << "***Error! Multiple integration methods are applied to the same particle" << endl << endl;
            throw std::runtime_error("Error adding integration method");
            }
        }
    
    // ensure that the method has a matching deltaT
    new_method->setDeltaT(m_deltaT);
    
    // add it to the list
    m_methods.push_back(new_method);
    }

/*! \post All integration methods are removed from this integrator
*/
void IntegratorTwoStep::removeAllIntegrationMethods()
    {
    m_methods.clear();
    m_gave_warning = false;
    }

/*! \returns true If all added integration methods have valid restart information
*/
bool IntegratorTwoStep::isValidRestart()
    {
    bool res = true;
    
    // loop through all methods
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // and them all together
        res = res && (*method)->isValidRestart();
        }
    return res;
    }

/*! \param group Group over which to count degrees of freedom.
    IntegratorTwoStep totals up the degrees of freedom that each integration method provide to the group.
    Three degrees of freedom are subtracted from the total to account for the constrained position of the system center of
    mass.
*/
unsigned int IntegratorTwoStep::getNDOF(boost::shared_ptr<ParticleGroup> group)
    {
    int res = 0;

    // loop through all methods
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // dd them all together
        res += (*method)->getNDOF(group);
        }
    
    return res - m_sysdef->getNDimensions() - getNDOFRemoved();
    }

/*! Compute accelerations if needed for the first step.
    If acceleration is available in the restart file, then just call computeNetForce so that net_force and net_virial
    are available for the logger. This solves ticket #393
*/
void IntegratorTwoStep::prepRun(unsigned int timestep)
    {
    // if we haven't been called before, then the net force and accelerations have not been set and we need to calculate them
    if (m_first_step)
        {
        m_first_step = false;
        m_prepared = true;
        
        // net force is always needed (ticket #393)
        computeNetForce(timestep);
        
        // but the accelerations only need to be calculated if the restart is not valid
        if (!isValidRestart())
            computeAccelerations(timestep);
        }
    }

/*! Return the combined flags of all integration methods.
*/
PDataFlags IntegratorTwoStep::getRequestedPDataFlags()
    {
    PDataFlags flags;

    // loop through all methods
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // or them all together
        flags |= (*method)->getRequestedPDataFlags();
        }

    return flags;
    }

void export_IntegratorTwoStep()
    {
    class_<IntegratorTwoStep, boost::shared_ptr<IntegratorTwoStep>, bases<Integrator>, boost::noncopyable>
        ("IntegratorTwoStep", init< boost::shared_ptr<SystemDefinition>, Scalar >())
        .def("addIntegrationMethod", &IntegratorTwoStep::addIntegrationMethod)
        .def("removeAllIntegrationMethods", &IntegratorTwoStep::removeAllIntegrationMethods)
        ;
    }

