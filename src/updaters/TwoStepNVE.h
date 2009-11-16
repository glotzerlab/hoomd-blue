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

#include "IntegrationMethodTwoStep.h"

#ifndef __TWO_STEP_NVE_H__
#define __TWO_STEP_NVE_H__

/*! \file TwoStepNVE.h
    \brief Declares the TwoStepNVE class
*/

//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface
    
    \ingroup updaters
*/
class TwoStepNVE : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNVE(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group);
        virtual ~TwoStepNVE() {};
        
        //! Sets the movement limit
        void setLimit(Scalar limit);
        
        //! Removes the limit
        void removeLimit();
        
        //! Sets the zero force option
        /*! \param zero_force Set to true to specify that the integration with a zero net force on each of the particles
                              in the group
        */
        void setZeroForce(bool zero_force) 
            {
            m_zero_force = zero_force;
            }
        
        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);
        
        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);
    
    protected:
        bool m_limit;       //!< True if we should limit the distance a particle moves in one step
        Scalar m_limit_val; //!< The maximum distance a particle is to move in one step
        bool m_zero_force;  //!< True if the integration step should ignore computed forces
    };

//! Exports the TwoStepNVE class to python
void export_TwoStepNVE();

#endif // #ifndef __TWO_STEP_NVE_H__

