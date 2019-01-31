// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ForceConstraint.h"
#include "hoomd/ParticleGroup.h"

#include <memory>

/*! \file OneDConstraint.h
    \brief Declares a class for computing 1D constraint forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ONE_D_CONSTRAINT_H__
#define __ONE_D_CONSTRAINT_H__

//! Applys a constraint force to prevent motion in x and y directions
/*! \ingroup computes
*/
class PYBIND11_EXPORT OneDConstraint : public ForceConstraint
    {
    public:
        //! Constructs the compute
        OneDConstraint(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         Scalar3 constraint_vec );

        //! Destructor
        virtual ~OneDConstraint();

        //! Set the force to a new value
        void setVector(Scalar3 constraint_vec);

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved();

    protected:
        std::shared_ptr<ParticleGroup> m_group;   //!< Group of particles on which this constraint is applied
        Scalar3 m_vec;  //!< The vector along which particles are constrained

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

    };

//! Exports the OneDConstraint class to python
void export_OneDConstraint(pybind11::module& m);

#endif
