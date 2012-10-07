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

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

IntegratorTwoStep::IntegratorTwoStep(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : Integrator(sysdef, deltaT), m_first_step(true), m_prepared(false), m_gave_warning(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing IntegratorTwoStep" << endl;
    }

IntegratorTwoStep::~IntegratorTwoStep()
    {
    m_exec_conf->msg->notice(5) << "Destroying IntegratorTwoStep" << endl;
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
        m_exec_conf->msg->warning() << "integrate.mode_standard: No integration methods are set, continuing anyways." << endl;
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

    // Update the rigid body particle positions and velocities if they are present
    if (m_sysdef->getRigidData()->getNumBodies() > 0)
        m_sysdef->getRigidData()->setRV(true);

    if (m_prof)
        m_prof->pop();

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if ghost atom positions are updated every time step
        m_comm->communicate(timestep+1);
        }
#endif

    // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);

    if (m_prof)
        m_prof->push("Integrate");

    // if the virial needs to be computed and there are rigid bodies, perform the virial correction
    PDataFlags flags = m_pdata->getFlags();
    if (flags[pdata_flag::isotropic_virial] && m_sysdef->getRigidData()->getNumBodies() > 0)
        m_sysdef->getRigidData()->computeVirialCorrectionStart();

    // perform the second step of the integration on all groups
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);

    // Update the rigid body particle velocities if they are present
    if (m_sysdef->getRigidData()->getNumBodies() > 0)
       m_sysdef->getRigidData()->setRV(false);

    // if the virial needs to be computed and there are rigid bodies, perform the virial correction
    if (flags[pdata_flag::isotropic_virial] && m_sysdef->getRigidData()->getNumBodies() > 0)
        m_sysdef->getRigidData()->computeVirialCorrectionEnd(m_deltaT/2.0);

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
        m_exec_conf->msg->warning() << "itegrate.mode_standard: An integration method has been added that operates on zero particles." << endl;
    
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        shared_ptr<ParticleGroup> intersection = ParticleGroup::groupIntersection(new_group, current_group);
        
        if (intersection->getNumMembers() > 0)
            {
            m_exec_conf->msg->error() << "itegrate.mode_standard: Multiple integration methods are applied to the same particle" << endl;
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
        
#ifdef ENABLE_MPI
    if (m_comm)
        m_comm->communicate(timestep);
#endif

        // net force is always needed (ticket #393)
        computeNetForce(timestep);
        
        // but the accelerations only need to be calculated if the restart is not valid
        if (!isValidRestart())
            computeAccelerations(timestep);
        
        // for the moment, isotropic_virial is invalid on the first step if there are any rigid bodies
        // a future update to the restart data format (that saves net_force and net_virial) will make it
        // valid when there is a valid restart
        if (m_sysdef->getRigidData()->getNumBodies() > 0)
            m_pdata->removeFlag(pdata_flag::isotropic_virial);
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

#ifdef ENABLE_MPI
//! Set the communicator to use
void IntegratorTwoStep::setCommunicator(boost::shared_ptr<Communicator> comm)
    {
    // set Communicator in all methods
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
            (*method)->setCommunicator(comm);

    Integrator::setCommunicator(comm);
    }
#endif

void export_IntegratorTwoStep()
    {
    class_<IntegratorTwoStep, boost::shared_ptr<IntegratorTwoStep>, bases<Integrator>, boost::noncopyable>
        ("IntegratorTwoStep", init< boost::shared_ptr<SystemDefinition>, Scalar >())
        .def("addIntegrationMethod", &IntegratorTwoStep::addIntegrationMethod)
        .def("removeAllIntegrationMethods", &IntegratorTwoStep::removeAllIntegrationMethods)
        ;
    }

