// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Initializers.h
    \brief Declares a few initializers for setting up ParticleData instances
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleData.h"

#ifndef __INITIALIZERS_H__
#define __INITIALIZERS_H__

//! Forward declaration of SnapshotSystemData
template <class Real> struct SnapshotSystemData;

//! Inits a ParticleData with a simple cubic array of particles
/*! A number of particles along each axis are specified along with a spacing
    between particles. This initializer only generates a single particle type.
    \ingroup data_structs
*/
class PYBIND11_EXPORT SimpleCubicInitializer
    {
    public:
        //! Set the parameters
        SimpleCubicInitializer(unsigned int M, Scalar spacing, const std::string &type_name);
        //! Empty Destructor
        virtual ~SimpleCubicInitializer() { }

        //! initializes a snapshot with the particle data
        virtual std::shared_ptr< SnapshotSystemData<Scalar> > getSnapshot() const;

    private:
        unsigned int m_M;   //!< Number of particles wide to make the box
        Scalar m_spacing;   //!< Spacing between particles
        BoxDim box;         //!< Precalculated box
        std::string m_type_name;    //!< Name of the particle type created
    };

//! Inits a ParticleData with randomly placed particles in a cube
/*! A minimum distance parameter is provided so that particles are never
    placed too close together. This initializer only generates a single particle
    type.
*/
class PYBIND11_EXPORT RandomInitializer
    {
    public:
        //! Set the parameters
        RandomInitializer(unsigned int N, Scalar phi_p, Scalar min_dist, const std::string &type_name);
        //! Empty Destructor
        virtual ~RandomInitializer() { }

        //! initializes a snapshot with the particle data
        virtual std::shared_ptr< SnapshotSystemData<Scalar> > getSnapshot() const;

        //! Sets the random seed to use in the generation
        void setSeed(unsigned int seed);

    protected:
        unsigned int m_N;           //!< Number of particles to generate
        Scalar m_phi_p;             //!< Packing fraction to generate the particles at
        Scalar m_min_dist;          //!< Minimum distance to separate particles by
        BoxDim m_box;               //!< Box to put the particles in
        std::string m_type_name;    //!< Name of the particle type created
    };

#endif
