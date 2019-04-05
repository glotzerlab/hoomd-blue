// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ?

#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include <memory>

/*! \file DynamicBond.h
    \brief Declares a class for computing bond breakage/formation
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

// #ifndef __CONSTRAINT_Ellipsoid_H__
// #define __CONSTRAINT_Ellipsoid_H__

//! Creates or breaks bonds with a given probability
/*!
*/
class PYBIND11_EXPORT DynamicBond : public Updater
    {
    public:
        //! Constructs the compute
        DynamicBond(std::shared_ptr<SystemDefinition> sysdef,
                 std::shared_ptr<ParticleGroup> group,
                 Scalar r_cut,
                 std::shared_ptr<NeighborList> nlist,
                 Scalar period,
                 // bond_type,
                 int seed,
                 Scalar prob_create,
                 Scalar prob_destroy);

        //! Destructor
        virtual ~DynamicBond();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    protected:
        std::shared_ptr<ParticleGroup> m_group;   //!< Group of particles to which the dynamic bonding is applied
        Scalar r_cut;          //!< cutoff radius
        std::shared_ptr<NeighborList> m_nlist;                 //!< neighborlist
        Scalar period;         //!< period to create/destroy bonds
        // bond_type;             //!< type of bond to be created or destroyed
        int seed;              //!< a seed for the random number generator
        Scalar prob_create;    //!< probability that a bond will be formed
        Scalar prob_destroy;   //!< probability that a bond will be destroyed
    private:
    };

//! Exports the DynamicBond class to python
void export_DynamicBond(pybind11::module& m);
