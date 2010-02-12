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
// Maintainer: ndtrung

#include <boost/shared_ptr.hpp>

#ifndef __FIRE_ENERGY_MINIMIZER_RIGID_H__
#define __FIRE_ENERGY_MINIMIZER_RIGID_H__

#include "FIREEnergyMinimizer.h"
#include "RigidData.h"

/*! \file FIREEnergyMinimizerRigid.h
    \brief Declares a class for energy minimization for rigid bodies 
*/

//! Finds the nearest basin in the potential energy landscape
/*! \b Overview
    
    \ingroup updaters
*/
class FIREEnergyMinimizerRigid : public FIREEnergyMinimizer
    {
    public:
        //! Constructs the minimizer and associates it with the system
        FIREEnergyMinimizerRigid(boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar, bool=true);
        virtual ~FIREEnergyMinimizerRigid() {}
        
        virtual void reset();
        
        //! Set the timestep
        virtual void setDeltaT(Scalar);

        //! Perform one minimization iteration
        virtual void update(unsigned int);

        //! Return whether or not the minimization has converged
        bool hasConverged() const {return m_converged;}

        //! Set the minimum number of steps for which the search direction must be bad before finding a new direction
        /*! \param nmin is the new nmin to set 
        */
        void setNmin(unsigned int nmin) {m_nmin = nmin;}
        
        //! Set the fractional increase in the timestep upon a valid search direction
        void setFinc(Scalar finc);
        
        //! Set the fractional increase in the timestep upon a valid search direction
        void setFdec(Scalar fdec);
        
        //! Set the relative strength of the coupling between the "f dot v" vs the "v" term 
        void setAlphaStart(Scalar alpha0);

        //! Set the fractional decrease in alpha upon finding a valid search direction 
        void setFalpha(Scalar falpha);
        
        //! Set the stopping criterion based on the total force on all particles in the system  
        /*! \param ftol is the new force tolerance to set
        */
        void setFtol(Scalar ftol) {m_ftol = ftol;}

        //! Set the stopping criterion based on the change in energy between successive iterations  
        /*! \param etol is the new energy tolerance to set
        */
        void setEtol(Scalar etol) {m_etol = etol;}
        
        //! Access the group
        boost::shared_ptr<ParticleGroup> getGroup() { return m_group; }       
        
        //! Get the period of minimization
        unsigned int getEvery() 
            { 
            return m_nevery; 
            }
            
        /*! Set the period of minimization
           \param nevery Period to set
        */
        void setEvery(unsigned int nevery) 
            { 
            m_nevery = nevery; 
            }
            
    protected:
        //! Function to create the underlying integrator 
        //virtual void createIntegrator();    
        
        boost::shared_ptr<RigidData> m_rigid_data;  //!< Pointer to the rigid data 
        unsigned int m_nparticles;                  //!< Total number of particles 
        unsigned int m_nevery;                      //!< Period of minimization
    private:
    
    };

//! Exports the FIREEnergyMinimizerRigid class to python
void export_FIREEnergyMinimizerRigid();

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_RIGID_H__

