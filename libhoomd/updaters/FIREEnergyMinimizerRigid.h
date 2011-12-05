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

// Maintainer: ndtrung

#include <boost/shared_ptr.hpp>

#ifndef __FIRE_ENERGY_MINIMIZER_RIGID_H__
#define __FIRE_ENERGY_MINIMIZER_RIGID_H__

#include "FIREEnergyMinimizer.h"
#include "RigidData.h"
#include "RigidBodyGroup.h"

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

        //! Perform one minimization iteration
        virtual void update(unsigned int);
        
        //! Access the group
        boost::shared_ptr<ParticleGroup> getGroup() { return m_group; }   
        
        //! Set the stopping criterion based on the total torque on all particles in the system  
        /*! \param wtol is the new torque tolerance to set
        */
        void setWtol(Scalar wtol) {m_wtol = wtol;}        
        
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
        boost::shared_ptr<RigidData> m_rigid_data;  //!< Pointer to the rigid data
        boost::shared_ptr<RigidBodyGroup> m_body_group; //!< Group of rigid bodies to work with
        unsigned int m_n_bodies;                    //!< Number of rigid bodies
        unsigned int m_nparticles;                  //!< Total number of particles 
        unsigned int m_nevery;                      //!< Period of minimization
        Scalar m_wtol;                          //!< stopping tolerance based on total torque        
    private:
    
    };

//! Exports the FIREEnergyMinimizerRigid class to python
void export_FIREEnergyMinimizerRigid();

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_RIGID_H__

