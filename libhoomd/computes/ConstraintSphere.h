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

#include "ForceConstraint.h"
#include "ParticleGroup.h"

/*! \file ConstraintSphere.h
    \brief Declares a class for computing sphere constraint forces
*/

#ifndef __CONSTRAINT_SPHERE_H__
#define __CONSTRAINT_SPHERE_H__

//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
*/
class ConstraintSphere : public ForceConstraint
    {
    public:
        //! Constructs the compute
        ConstraintSphere(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<ParticleGroup> group,
                         Scalar3 P,
                         Scalar r);
        
        //! Set the force to a new value
        void setSphere(Scalar3 P, Scalar r);

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved();

    protected:
        boost::shared_ptr<ParticleGroup> m_group;   //!< Group of particles on which this constraint is applied
        Scalar3 m_P;         //!< Position of the sphere
        Scalar m_r;          //!< Radius of the sphere
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    
    private:
        //! Validate that the sphere is in the box and all particles are very near the constraint
        void validate();
    };

//! Exports the ConstraintSphere class to python
void export_ConstraintSphere();

#endif

