/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#include <boost/shared_ptr.hpp>

#include "Compute.h"

/*! \file TempCompute.h
    \brief Declares a class for computing temperatures
*/

#ifndef __TEMPCOMPUTE_H__
#define __TEMPCOMPUTE_H__

//! Computes the temperature of the particle system
/*! An instantaneous temperature is computed in the standard method: AVG Kinetic energy = dof/2 * k_B * T
    The number of degrees of freedom defaults to 3*N, but can be changed with setDOF().
    \ingroup computes
*/
class TempCompute : public Compute
    {
    public:
        //! Constructs the compute
        TempCompute(boost::shared_ptr<SystemDefinition> sysdef);
        
        //! Compute the temperature
        virtual void compute(unsigned int timestep);
        
        //! Change the number of degrees of freedom
        /*! \param dof Number of degrees of freedom to set
        */
        void setDOF(unsigned int dof)
            {
            m_dof = dof;
            }
            
        //! Returns the temperature last computed by compute()
        /*! \returns Instantaneous temperature of the system
        */
        Scalar getTemp()
            {
            return m_temp;
            }
    protected:
        Scalar m_temp;  //!< Stores the last computed value of the temperature
        unsigned int m_dof; //!< Stores the number of degrees of freedom in the system
        
        //! Does the actual computation
        void computeTemp();
    };

//! Exports the TempCompute class to python
void export_TempCompute();

#endif
