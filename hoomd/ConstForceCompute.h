// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ForceCompute.h"
#include "ParticleGroup.h"

#include <boost/shared_ptr.hpp>

/*! \file ConstForceCompute.h
    \brief Declares a class for computing constant forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

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

        //! Destructor
        ~ConstForceCompute();

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

        Scalar m_fx; //!< Constant force in x-direction
        Scalar m_fy; //!< Constant force in y-direction
        Scalar m_fz; //!< Constant force in z-direction

        //! Group of particles to apply force to
        boost::shared_ptr<ParticleGroup> m_group;
    };

//! Exports the ConstForceComputeClass to python
void export_ConstForceCompute();

#endif
