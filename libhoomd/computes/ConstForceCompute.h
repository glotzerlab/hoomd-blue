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

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "ParticleGroup.h"

/*! \file ConstForceCompute.h
    \brief Declares a class for computing constant forces
*/

#ifndef __CONSTFORCECOMPUTE_H__
#define __CONSTFORCECOMPUTE_H__

//! Adds a constant force to a number of particles
/*! \ingroup computes
*/
class ConstForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        ConstForceCompute(boost::shared_ptr<SystemDefinition> sysdef, Scalar fx, Scalar fy, Scalar fz);
        ConstForceCompute(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz);

        //! Set the force to a new value
        void setForce(Scalar fx, Scalar fy, Scalar fz);
        
        //! Set the force for an individual particle
        void setParticleForce(unsigned int i, Scalar fx, Scalar fy, Scalar fz);
        
        //! Set force for a particle group
        void setGroupForce(boost::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz);

    protected:

        //! Function that is called on every particle sort
        void rearrangeForces();

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

    private:

        // Constant force values 
        Scalar m_fx; //! Constant force in x-direction
        Scalar m_fy; //! Constant force in y-direction
        Scalar m_fz; //! Constant force in z-direction

        //! Group of particles to apply force to
        boost::shared_ptr<ParticleGroup> m_group; 
    };

//! Exports the ConstForceComputeClass to python
void export_ConstForceCompute();

#endif

