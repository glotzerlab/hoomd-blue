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
#include "Variant.h"
#include "ComputeThermo.h"

#ifndef __TWO_STEP_NPT_H__
#define __TWO_STEP_NPT_H__

/*! \file TwoStepNPT.h
    \brief Declares the TwoStepNPT class
*/

//! Integrates part of the system forward in two steps in the NPT ensemble
/*! Implements Nose-Hoover/Anderson NPT integration through the IntegrationMethodTwoStep interface
    
    Integrator variables mapping:
     - [0] -> xi
     - [1] -> eta

    The instantaneous temperature of the system is computed with the provided ComputeThermo. Correct dynamics require
    that the thermo computes the temperature of the assigned group and with D*N-D degrees of freedom. TwoStepNPT does
    not check for these conditions. In addition, TwoStepNPT needs a ComputeThermo that operates on the group of all
    particles.

    \ingroup updaters
*/
class TwoStepNPT : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNPT(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
                   boost::shared_ptr<ComputeThermo> thermo_group,
                   boost::shared_ptr<ComputeThermo> thermo_all,
                   Scalar tau,
                   Scalar tauP,
                   boost::shared_ptr<Variant> T,
                   boost::shared_ptr<Variant> P);
        virtual ~TwoStepNPT() {};
        
        //! Update the temperature
        /*! \param T New temperature to set
        */
        virtual void setT(boost::shared_ptr<Variant> T)
            {
            m_T = T;
            }
        
        //! Update the pressure
        /*! \param P New pressure to set
        */
        virtual void setP(boost::shared_ptr<Variant> P)
            {
            m_P = P;
            }
        
        //! Update the tau value
        /*! \param tau New time constant to set
        */
        virtual void setTau(Scalar tau)
            {
            m_tau = tau;
            }
        
        //! Update the nuP value
        /*! \param tauP New pressure constant to set
        */
        virtual void setTauP(Scalar tauP)
            {
            m_tauP = tauP;
            }
        
        //! Set the partial scale option
        /*! \param partial_scale New partial_scale option to set
        */
        void setPartialScale(bool partial_scale)
            {
            m_partial_scale = partial_scale;
            }
        
        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);
        
        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);
    
    protected:
        boost::shared_ptr<ComputeThermo> m_thermo_group;   //!< ComputeThermo operating on the integrated group
        boost::shared_ptr<ComputeThermo> m_thermo_all;     //!< ComputeThermo operating on the group of all particles
        
        bool m_partial_scale;           //!< True if only the particles in the group should be scaled to the new box
        Scalar m_tau;                   //!< tau value for Nose-Hoover
        Scalar m_tauP;                  //!< tauP value for the barostat
        boost::shared_ptr<Variant> m_T; //!< Temperature set point
        boost::shared_ptr<Variant> m_P; //!< Pressure set point
        Scalar m_curr_group_T;          //!< Current group temperature
        Scalar m_curr_P;                //!< Current system pressure
        Scalar m_V;                     //!< Current volume
        Scalar m_Lx;                    //!< Box length in x direction
        Scalar m_Ly;                    //!< Box length in y direction
        Scalar m_Lz;                    //!< Box length in z direction
    };

//! Exports the TwoStepNVT class to python
void export_TwoStepNPT();

#endif // #ifndef __TWO_STEP_NPT_H__

