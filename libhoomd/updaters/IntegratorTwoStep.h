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

// Maintainer: joaander

#include "Integrator.h"
#include "IntegrationMethodTwoStep.h"

#ifndef __INTEGRATOR_TWO_STEP_H__
#define __INTEGRATOR_TWO_STEP_H__

/*! \file IntegratorTwoStep.h
    \brief Declares an integrator for performing two-step integration on multiple groups
*/

//! Integrates the system forward one step with possibly multiple methods
/*! See IntegrationMethodTwoStep for most of the design notes regarding group integration. IntegratorTwoStep merely
    implements most of the things discussed there.
    
    Notable design elements:
    - setDeltaT results in deltaT being set on all current integration methods
    - to ensure that new methods also get set, addIntegrationMethod() also calls setDeltaT on the method
    - to interface with the python script, a removeAllIntegrationMethods() method is provided to clear the list so they
      can be cleared and re-added from hoomd_script's internal list
    
    To ensure that the user does not make a mistake and specify more than one method operating on a single particle,
    the particle groups are checked for intersections whenever a new method is added in addIntegrationMethod()
    
    \ingroup updaters
*/
class IntegratorTwoStep : public Integrator
    {
    public:
        //! Constructor
        IntegratorTwoStep(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);
        
        //! Destructor
        virtual ~IntegratorTwoStep();
        
        //! Sets the profiler for the compute to use
        virtual void setProfiler(boost::shared_ptr<Profiler> prof);
        
        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
                
        //! Returns logged values
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
                
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
        
        //! Change the timestep
        virtual void setDeltaT(Scalar deltaT);
        
        //! Add a new integration method to the list that will be run
        virtual void addIntegrationMethod(boost::shared_ptr<IntegrationMethodTwoStep> new_method);
        
        //! Remove all integration methods
        virtual void removeAllIntegrationMethods();
        
        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getNDOF(boost::shared_ptr<ParticleGroup> group);

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep);

        //! Get needed pdata flags
        virtual PDataFlags getRequestedPDataFlags();

    protected:
        //! Helper method to test if all added methods have valid restart information
        bool isValidRestart();

        std::vector< boost::shared_ptr<IntegrationMethodTwoStep> > m_methods;   //!< List of all the integration methods
        
        bool m_first_step;      //!< True before the first call to update()
        bool m_prepared;        //!< True if preprun has been called
        bool m_gave_warning;    //!< True if a warning has been given about no methods added
    
    };

//! Exports the IntegratorTwoStep class to python
void export_IntegratorTwoStep();

#endif // #ifndef __INTEGRATOR_TWO_STEP_H__

