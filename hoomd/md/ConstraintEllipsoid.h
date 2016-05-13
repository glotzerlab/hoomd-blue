// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include <boost/shared_ptr.hpp>

/*! \file ConstraintEllipsoid.h
    \brief Declares a class for computing ellipsoid constraint forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __CONSTRAINT_Ellipsoid_H__
#define __CONSTRAINT_Ellipsoid_H__

//! Applys a constraint force to keep a group of particles on a Ellipsoid
/*! \ingroup computes
*/
class ConstraintEllipsoid : public Updater
    {
    public:
        //! Constructs the compute
        ConstraintEllipsoid(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<ParticleGroup> group,
                         Scalar3 P,
                         Scalar rx,
                         Scalar ry,
                         Scalar rz);

        //! Destructor
        virtual ~ConstraintEllipsoid();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    protected:
        boost::shared_ptr<ParticleGroup> m_group;   //!< Group of particles on which this constraint is applied
        Scalar3 m_P;          //!< Position of the Ellipsoid
        Scalar m_rx;          //!< Radius in X direction of the Ellipsoid
        Scalar m_ry;          //!< Radius in Y direction of the Ellipsoid
        Scalar m_rz;          //!< Radius in Z direction of the Ellipsoid

    private:
        //! Validate that the ellipsoid is in the box and all particles are very near the constraint
        void validate();
    };

//! Exports the ConstraintEllipsoid class to python
void export_ConstraintEllipsoid();

#endif
