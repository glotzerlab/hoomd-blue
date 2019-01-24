// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ForceCompute.h"
#include "hoomd/ParticleGroup.h"
#include <memory>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "EvaluatorConstraintEllipsoid.h"


/*! \file ActiveForceCompute.h
    \brief Declares a class for computing active forces and torques
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ACTIVEFORCECOMPUTE_H__
#define __ACTIVEFORCECOMPUTE_H__

//! Adds an active force to a number of particles
/*! \ingroup computes
*/
class PYBIND11_EXPORT ActiveForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        ActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             int seed, pybind11::list f_lst, pybind11::list t_lst,
                             bool orientation_link, bool orientation_reverse_link,
                             Scalar rotation_diff,
                             Scalar3 P,
                             Scalar rx,
                             Scalar ry,
                             Scalar rz);

        //! Destructor
        ~ActiveForceCompute();

    protected:
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Set forces for particles
        virtual void setForces();

        //! Orientational diffusion for spherical particles
        virtual void rotationalDiffusion(unsigned int timestep);

        //! Set constraints if particles confined to a surface
        virtual void setConstraint();

        std::shared_ptr<ParticleGroup> m_group;   //!< Group of particles on which this force is applied
        bool m_orientationLink;
        bool m_orientationReverseLink;
        Scalar m_rotationDiff;
        Scalar m_rotationConst;
        Scalar3 m_P;          //!< Position of the Ellipsoid
        Scalar m_rx;          //!< Radius in X direction of the Ellipsoid
        Scalar m_ry;          //!< Radius in Y direction of the Ellipsoid
        Scalar m_rz;          //!< Radius in Z direction of the Ellipsoid
        int m_seed;           //!< Random number seed
        GPUArray<Scalar3> m_f_activeVec; //! active force unit vectors for each particle
        GPUArray<Scalar> m_f_activeMag; //! active force magnitude for each particle

        GPUArray<Scalar3> m_t_activeVec; //! active torque unit vectors for each particle
        GPUArray<Scalar> m_t_activeMag; //! active torque magnitude for each particle

        unsigned int last_computed;
    };

//! Exports the ActiveForceComputeClass to python
void export_ActiveForceCompute(pybind11::module& m);
#endif
