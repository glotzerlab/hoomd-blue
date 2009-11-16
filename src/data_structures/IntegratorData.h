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

// $Id: WallData.h 2262 2009-10-30 06:59:14Z askeys $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/branches/binary-restart/src/data_structures/WallData.h $
// Maintainer: joaander

/*! \file WallData.h
    \brief Contains declarations for WallData.
 */

#ifndef __INTEGRATORDATA_H__
#define __INTEGRATORDATA_H__

#include "ParticleData.h"
#include <string>


//! Stores integrator variables
/*! The integration state is necessary for exact restarts.  Extended systems 
    integrators in the spirit of Nose-Hoover store the positions, velocities, 
    etc. of the fictitious variables.  Other integrators store a random number 
    seed.
    \ingroup data_structs
*/
struct IntegratorVariables
    {
    std::string type;                   //!<The type of integrator (NVT, NPT, etc.)
    std::vector<Scalar> variable;       //!<Variables that define the integration state
    };

//! Stores all integrator variables in the simulation
/*! IntegratorData keeps track of the parameters for all of the integrators 
    defined in the simulation. 
    
    \ingroup data_structs
*/
class IntegratorData
    {
    public:
        //! Constructs an empty list with no integrator variables
        IntegratorData() {}
        
        //! Destructor
        ~IntegratorData() {}
        
        //! Register an integrator (should occur during integrator construction)
        void registerIntegrator(unsigned int);
        
        //! Get the number of integrator variables
        /*! \return Number of integrator variables present
        */
        const unsigned int getNumIntegrators() const
            {
            return (unsigned int)m_integrator_variables.size();
            }
            
        //! Get a collection of integrator variables
        /*! \param i access integrator variables for integrator i
        */
        const IntegratorVariables& getIntegratorVariables(unsigned int i) const
            {
            assert(i < m_integrator_variables.size()); return m_integrator_variables[i];
            }
            
        //! Set a collection of integrator variables
        /*! \param i set integrator variables for integrator i
        */
        void setIntegratorVariables(int i, const IntegratorVariables& v)
            {
            assert(i < m_integrator_variables.size()); m_integrator_variables[i] = v;
            }
            
    private:
    
        std::vector<IntegratorVariables> m_integrator_variables;
        
    };

//! Exports IntegratorData to python
void export_IntegratorData();

#endif

