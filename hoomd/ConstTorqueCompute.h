// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ForceCompute.h"
#include "ParticleGroup.h"

#include <memory>

/*! \file ConstTorqueCompute.h
    \brief Declares a class for computing constant forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CONSTTORQUECOMPUTE_H__
#define __CONSTTORQUECOMPUTE_H__

//! Adds a constant force to a number of particles
/*! \ingroup computes
*/
class ConstTorqueCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        ConstTorqueCompute(std::shared_ptr<SystemDefinition> sysdef, Scalar fx, Scalar fy, Scalar fz);
        ConstTorqueCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz);

        //! Destructor
        ~ConstTorqueCompute();

        //! Set the force to a new value
        void setTorque(Scalar fx, Scalar fy, Scalar fz);

        //! Set the force for an individual particle
        void setParticleTorque(unsigned int i, Scalar fx, Scalar fy, Scalar fz);

        //! Set force for a particle group
        void setGroupTorque(std::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz);

    protected:

        //! Function that is called on every particle sort
        void rearrangeTorques();

        //! Actually compute the forces
        virtual void computeTorques(unsigned int timestep);

    private:

        Scalar m_fx; //!< Constant force in x-direction
        Scalar m_fy; //!< Constant force in y-direction
        Scalar m_fz; //!< Constant force in z-direction

        //! Group of particles to apply force to
        std::shared_ptr<ParticleGroup> m_group;
    };

//! Exports the ConstTorqueComputeClass to python
void export_ConstTorqueCompute(pybind11::module& m);

#endif
